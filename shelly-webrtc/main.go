// shelly-webrtc-grab: Connects to a Shelly camera via WebRTC, receives H.264
// RTP, reassembles NAL units and writes Annex-B byte stream to stdout. FFmpeg
// reads it on stdin: `ffmpeg -f h264 -r 25 -i pipe:0 …`.
//
// Three handshake variants are supported (auto-detected at runtime, in order):
//
//  • WHEP (POST /camera/0/whep/0, application/sdp)
//      The endpoint Shelly added in firmware ≥ 2.1.99-dev (June 2026).
//      RFC 9725-style: raw SDP offer in request body, raw SDP answer in the
//      2xx response body. Stream id 0 = main 1920×1080, 1 = sub 640×360.
//
//  • CLIENT-OFFER (POST /rpc/Streamer.GetAnswer, JSON-wrapped SDP)
//      Older Shelly Plus firmware (~2.0). JSON body with `sdp` + `ice_servers`
//      + `stream_id`. Returns JSON `{sdp:"…"}` (occasionally double-wrapped).
//
//  • CAMERA-OFFER (POST /rpc/Streamer.Offer + /rpc/Streamer.Answer)
//      Legacy Shelly Plus firmware (≤ ~1.x). Camera makes the offer first,
//      we send the answer.
//
// Usage: shelly-webrtc-grab <camera-base-url>  (e.g. http://192.168.3.227)
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/pion/interceptor"
	"github.com/pion/rtp/codecs"
	"github.com/pion/webrtc/v3"
)

const (
	streamerOffer     = "/rpc/Streamer.Offer"     // legacy (camera-offer)
	streamerAnswer    = "/rpc/Streamer.Answer"    // legacy (camera-offer)
	streamerGetAnswer = "/rpc/Streamer.GetAnswer" // intermediate (client-offer JSON)
	whepEndpoint      = "/camera/0/whep/0"        // modern (WHEP, application/sdp)
)

// Same TURN/STUN list that the official Shelly browser client uses — the
// Streamer.GetAnswer flow needs them in the JSON body. WHEP doesn't need them
// in the body (raw SDP) but we keep them in the PC config for ICE gathering
// when the camera and our host are on different subnets.
var shellyICEServers = []webrtc.ICEServer{
	{URLs: []string{"turn:turn.shelly.cloud:3478"}, Username: "admin2", Credential: "admin2"},
	{URLs: []string{"stun:stun.shelly.cloud:3478"}},
}
var shellyICEServersWire = []string{
	"turn://admin2:admin2@turn.shelly.cloud:3478",
	"stun://stun.shelly.cloud:3478",
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: shelly-webrtc-grab <camera-base-url>")
		os.Exit(1)
	}
	baseURL := strings.TrimRight(os.Args[1], "/")
	log.SetOutput(os.Stderr)
	log.SetFlags(log.Ltime)

	// Когато recorder-ът ни kill-ва (SIGTERM / SIGINT), направи DELETE на
	// активния WHEP resource за да не остане „заето" място на камерата
	// (Shelly Plus има само ~1-2 паралелни WebRTC сесии).
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-sig
		log.Printf("[webrtc] received signal, cleaning up WHEP resource")
		whepCleanup()
		os.Exit(0)
	}()

	for {
		if err := run(baseURL); err != nil {
			log.Printf("[webrtc] error: %v — retry in 5s", err)
			whepCleanup()
			time.Sleep(5 * time.Second)
		}
	}
}

func run(baseURL string) error {
	pc, done, out, err := newPeerConnection()
	if err != nil {
		return err
	}
	defer pc.Close()

	// Try modern WHEP first → Streamer.GetAnswer → legacy Streamer.Offer.
	if whepErr := whepHandshake(baseURL, pc); whepErr != nil {
		log.Printf("[webrtc] WHEP failed: %v — trying Streamer.GetAnswer", whepErr)
		// PC has client-offer set (good for both fallbacks since they also
		// expect a client offer / answer pair, except legacy which needs a
		// fresh PC to receive the camera's offer).
		if jsonErr := clientOfferJSON(baseURL, pc); jsonErr != nil {
			log.Printf("[webrtc] Streamer.GetAnswer failed: %v — trying legacy Streamer.Offer", jsonErr)
			pc.Close()
			pc, done, out, err = newPeerConnection()
			if err != nil {
				return err
			}
			defer pc.Close()
			if legacyErr := cameraOffer(baseURL, pc); legacyErr != nil {
				return fmt.Errorf("all flows failed; whep=%v json=%v legacy=%v",
					whepErr, jsonErr, legacyErr)
			}
		}
	}

	res := <-done
	out.Flush()
	whepCleanup()
	return res
}

// newPeerConnection sets up pion media + RTP→Annex-B writer. Caller must
// later call CreateOffer / SetLocalDescription before sending to the camera.
func newPeerConnection() (*webrtc.PeerConnection, chan error, *bufio.Writer, error) {
	m := &webrtc.MediaEngine{}
	if err := m.RegisterDefaultCodecs(); err != nil {
		return nil, nil, nil, err
	}
	ir := &interceptor.Registry{}
	if err := webrtc.RegisterDefaultInterceptors(m, ir); err != nil {
		return nil, nil, nil, err
	}
	api := webrtc.NewAPI(
		webrtc.WithMediaEngine(m),
		webrtc.WithInterceptorRegistry(ir),
	)
	pc, err := api.NewPeerConnection(webrtc.Configuration{
		ICEServers:   shellyICEServers,
		BundlePolicy: webrtc.BundlePolicyMaxBundle,
	})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("new peer connection: %w", err)
	}

	if _, err := pc.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
		return nil, nil, nil, err
	}
	if _, err := pc.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
		return nil, nil, nil, err
	}

	done := make(chan error, 4)
	out := bufio.NewWriterSize(os.Stdout, 2<<20)

	pc.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		if track.Kind() != webrtc.RTPCodecTypeVideo {
			go func() {
				buf := make([]byte, 1500)
				for {
					if _, _, err := track.Read(buf); err != nil {
						return
					}
				}
			}()
			return
		}
		log.Printf("[webrtc] video track: %s pt=%d", track.Codec().MimeType, track.PayloadType())
		h264 := &codecs.H264Packet{}
		// Gate-ваме output-а до първия SPS (NAL type 7). Pion връща RTP
		// пакетите от средата на GOP-а → ако пуснем суров поток без SPS/PPS+IDR
		// в началото, ffmpeg `-f h264` се проваля с "Invalid data found"
		// (AVERROR_INVALIDDATA → exit 183) и recorder-ът влиза в restart loop.
		// Изчакваме clean stream start, после пишем всичко.
		started := false
		for {
			pkt, _, err := track.ReadRTP()
			if err != nil {
				select {
				case done <- fmt.Errorf("ReadRTP: %w", err):
				default:
				}
				return
			}
			nalData, err := h264.Unmarshal(pkt.Payload)
			if err != nil || len(nalData) == 0 {
				continue
			}
			if !started {
				if !containsSPS(nalData) {
					continue // пропускай докато не дойде SPS
				}
				started = true
				log.Printf("[webrtc] SPS получен — започвам да пиша stream")
			}
			out.Write(nalData)
			if pkt.Marker {
				if err := out.Flush(); err != nil {
					select {
					case done <- fmt.Errorf("stdout flush: %w", err):
					default:
					}
					return
				}
			}
		}
	})

	pc.OnICEConnectionStateChange(func(s webrtc.ICEConnectionState) {
		log.Printf("[webrtc] ICE: %s", s)
		switch s {
		case webrtc.ICEConnectionStateFailed,
			webrtc.ICEConnectionStateDisconnected,
			webrtc.ICEConnectionStateClosed:
			select {
			case done <- fmt.Errorf("ICE state: %s", s):
			default:
			}
		}
	})

	return pc, done, out, nil
}

// whepResource is the absolute URL the WHEP server returned in `Location`.
// On graceful exit we DELETE it so the camera releases the WebRTC resource.
// Without this, repeated runs against the same camera can pile up sessions
// that bump our new handshake off after a couple of seconds (Shelly cameras
// have a limited number of concurrent WebRTC peers).
var whepResource string

// whepHandshake — RFC 9725-style: raw SDP offer in body, raw SDP answer in 2xx.
// Shelly added this endpoint in firmware ≥ 2.1.99-dev (June 2026) and it
// replaces the older Streamer.GetAnswer / Streamer.Offer RPCs (both of which
// the same firmware breaks).
func whepHandshake(baseURL string, pc *webrtc.PeerConnection) error {
	offer, err := pc.CreateOffer(nil)
	if err != nil {
		return fmt.Errorf("create offer: %w", err)
	}
	if err := pc.SetLocalDescription(offer); err != nil {
		return fmt.Errorf("set local description: %w", err)
	}
	gatherDone := webrtc.GatheringCompletePromise(pc)
	select {
	case <-gatherDone:
	case <-time.After(10 * time.Second):
		return fmt.Errorf("ICE gather timeout")
	}

	url := baseURL + whepEndpoint
	req, err := http.NewRequest("POST", url, strings.NewReader(pc.LocalDescription().SDP))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/sdp")
	req.Header.Set("Accept", "application/sdp")
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return fmt.Errorf("WHEP POST: %w", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("WHEP HTTP %d: %s", resp.StatusCode, truncate(body, 200))
	}
	if !strings.HasPrefix(strings.TrimSpace(string(body)), "v=") {
		return fmt.Errorf("WHEP response is not SDP: %s", truncate(body, 200))
	}
	loc := resp.Header.Get("Location")
	if loc != "" {
		// Camera returned a relative path like "/camera/0/whep/0/XXXX".
		if strings.HasPrefix(loc, "/") {
			whepResource = baseURL + loc
		} else {
			whepResource = loc
		}
	}
	log.Printf("[webrtc] WHEP %d %s, location=%s, answer %d bytes",
		resp.StatusCode, resp.Header.Get("Content-Type"), loc, len(body))

	if err := pc.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeAnswer,
		SDP:  string(body),
	}); err != nil {
		return fmt.Errorf("set remote description: %w", err)
	}
	log.Printf("[webrtc] WHEP handshake complete")
	return nil
}

// whepCleanup — best-effort DELETE of the WHEP resource so the camera frees
// the slot. Quiet on failure (camera may already have GC'd it).
func whepCleanup() {
	if whepResource == "" {
		return
	}
	req, err := http.NewRequest("DELETE", whepResource, nil)
	if err != nil {
		return
	}
	resp, err := (&http.Client{Timeout: 3 * time.Second}).Do(req)
	if err == nil {
		resp.Body.Close()
		log.Printf("[webrtc] WHEP DELETE %s → %d", whepResource, resp.StatusCode)
	}
	whepResource = ""
}

// clientOfferJSON — the intermediate Shelly Plus firmware flow.
// Body: { sdp, ice_servers, stream_id }. Response: { sdp } (sometimes nested).
func clientOfferJSON(baseURL string, pc *webrtc.PeerConnection) error {
	// SDP from previous WHIP attempt is reusable; ICE has already gathered.
	if pc.LocalDescription() == nil {
		offer, err := pc.CreateOffer(nil)
		if err != nil {
			return fmt.Errorf("create offer: %w", err)
		}
		if err := pc.SetLocalDescription(offer); err != nil {
			return fmt.Errorf("set local description: %w", err)
		}
		<-webrtc.GatheringCompletePromise(pc)
	}

	reqBody, _ := json.Marshal(map[string]any{
		"sdp":         pc.LocalDescription().SDP,
		"ice_servers": shellyICEServersWire,
		"stream_id":   0,
	})
	resp, err := http.Post(baseURL+streamerGetAnswer, "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("GetAnswer request: %w", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("GetAnswer HTTP %d: %s", resp.StatusCode, truncate(body, 200))
	}
	if len(bytes.TrimSpace(body)) == 0 {
		return fmt.Errorf("GetAnswer empty body")
	}

	var wrap struct {
		Sdp string `json:"sdp"`
	}
	if err := json.Unmarshal(body, &wrap); err != nil || wrap.Sdp == "" {
		return fmt.Errorf("parse GetAnswer: %w body=%s", err, truncate(body, 200))
	}
	answerSdp := wrap.Sdp
	if strings.HasPrefix(strings.TrimSpace(answerSdp), "{") {
		var nested struct {
			Sdp string `json:"sdp"`
		}
		if err := json.Unmarshal([]byte(answerSdp), &nested); err == nil && nested.Sdp != "" {
			answerSdp = nested.Sdp
		}
	}

	if err := pc.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeAnswer,
		SDP:  answerSdp,
	}); err != nil {
		return fmt.Errorf("set remote description (answer): %w", err)
	}
	log.Printf("[webrtc] Streamer.GetAnswer handshake complete")
	return nil
}

// cameraOffer — legacy: camera makes the SDP offer, we answer.
func cameraOffer(baseURL string, pc *webrtc.PeerConnection) error {
	resp, err := http.Post(baseURL+streamerOffer, "application/json", bytes.NewBufferString("{}"))
	if err != nil {
		return fmt.Errorf("offer request: %w", err)
	}
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return fmt.Errorf("read offer body: %w", err)
	}
	if len(bytes.TrimSpace(body)) == 0 {
		return fmt.Errorf("Streamer.Offer empty body")
	}
	var offerResp struct {
		SDP       string `json:"sdp"`
		SessionID string `json:"session_id"`
	}
	if err := json.Unmarshal(body, &offerResp); err != nil {
		return fmt.Errorf("parse offer: %w body=%s", err, truncate(body, 200))
	}

	if err := pc.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeOffer,
		SDP:  offerResp.SDP,
	}); err != nil {
		return fmt.Errorf("set remote description (offer): %w", err)
	}

	gatherDone := webrtc.GatheringCompletePromise(pc)
	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		return fmt.Errorf("create answer: %w", err)
	}
	if err := pc.SetLocalDescription(answer); err != nil {
		return fmt.Errorf("set local description: %w", err)
	}
	select {
	case <-gatherDone:
	case <-time.After(10 * time.Second):
		return fmt.Errorf("ICE gather timeout")
	}

	ansPayload, _ := json.Marshal(map[string]string{
		"session_id": offerResp.SessionID,
		"sdp":        pc.LocalDescription().SDP,
	})
	ans, err := http.Post(baseURL+streamerAnswer, "application/json", bytes.NewReader(ansPayload))
	if err != nil {
		return fmt.Errorf("send answer: %w", err)
	}
	ansBody, _ := io.ReadAll(ans.Body)
	ans.Body.Close()
	log.Printf("[webrtc] legacy answer sent, camera replied: %s", truncate(ansBody, 160))
	return nil
}

// containsSPS сканира Annex-B буфер (NAL units с 00000001 / 000001 start
// codes — както ги връща pion codecs.H264Packet) за SPS NAL (type 7).
// SPS носи геометрията/профила на потока; ffmpeg има нужда от него (заедно с
// PPS+IDR, които Shelly праща веднага след SPS) за да отвори input-а.
func containsSPS(b []byte) bool {
	i := 0
	for i+3 < len(b) {
		// open 3- или 4-байтов start code
		if b[i] == 0x00 && b[i+1] == 0x00 {
			var nalStart int
			if b[i+2] == 0x01 {
				nalStart = i + 3
			} else if i+4 < len(b) && b[i+2] == 0x00 && b[i+3] == 0x01 {
				nalStart = i + 4
			} else {
				i++
				continue
			}
			if nalStart < len(b) && (b[nalStart]&0x1F) == 7 {
				return true
			}
			i = nalStart
			continue
		}
		i++
	}
	return false
}

func truncate(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[:n]) + "…"
}
