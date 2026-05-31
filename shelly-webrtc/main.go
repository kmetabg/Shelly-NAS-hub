// shelly-webrtc-grab: Connects to a Shelly camera via WebRTC (Streamer.Offer / Streamer.Answer RPC),
// receives H.264 RTP, reassembles NAL units and writes Annex-B stream to stdout.
// ffmpeg reads from stdin: ffmpeg -f h264 -r 25 -i pipe:0 ...
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
	"time"

	"github.com/pion/interceptor"
	"github.com/pion/rtp/codecs"
	"github.com/pion/webrtc/v3"
)

type offerResp struct {
	SDP       string `json:"sdp"`
	SessionID string `json:"session_id"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: shelly-webrtc-grab <camera-base-url>")
		os.Exit(1)
	}
	baseURL := os.Args[1]
	log.SetOutput(os.Stderr)
	log.SetFlags(log.Ltime)

	for {
		if err := run(baseURL); err != nil {
			log.Printf("[webrtc] error: %v — retry in 5s", err)
			time.Sleep(5 * time.Second)
		}
	}
}

func run(baseURL string) error {
	// ── 1. GET SDP offer from camera ────────────────────────────────────────
	resp, err := http.Post(baseURL+"/rpc/Streamer.Offer",
		"application/json", bytes.NewBufferString("{}"))
	if err != nil {
		return fmt.Errorf("offer request: %w", err)
	}
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return fmt.Errorf("read offer body: %w", err)
	}

	var offer offerResp
	if err := json.Unmarshal(body, &offer); err != nil {
		return fmt.Errorf("parse offer: %w, body=%s", err, body)
	}
	log.Printf("[webrtc] offer received, session=%s", offer.SessionID)

	// ── 2. Create pion PeerConnection ───────────────────────────────────────
	m := &webrtc.MediaEngine{}
	if err := m.RegisterDefaultCodecs(); err != nil {
		return err
	}
	ir := &interceptor.Registry{}
	if err := webrtc.RegisterDefaultInterceptors(m, ir); err != nil {
		return err
	}
	api := webrtc.NewAPI(
		webrtc.WithMediaEngine(m),
		webrtc.WithInterceptorRegistry(ir),
	)
	pc, err := api.NewPeerConnection(webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{}, // LAN only — no STUN/TURN needed
	})
	if err != nil {
		return fmt.Errorf("new peer connection: %w", err)
	}
	defer pc.Close()

	// Declare we want to receive video and audio (camera is sendonly)
	if _, err = pc.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
		return err
	}
	if _, err = pc.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
		return err
	}

	done := make(chan error, 2)
	out := bufio.NewWriterSize(os.Stdout, 2<<20) // 2 MB output buffer

	pc.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		if track.Kind() != webrtc.RTPCodecTypeVideo {
			// Drain audio silently so the camera doesn't stall
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

		for {
			pkt, _, err := track.ReadRTP()
			if err != nil {
				select {
				case done <- fmt.Errorf("ReadRTP: %w", err):
				default:
				}
				return
			}

			// Pion's H264Packet.Unmarshal already PREPENDS the Annex-B start code
			// (0x00000001) to each returned NAL unit. For STAP-A it returns multiple
			// NALs concatenated with start codes. For FU-A it buffers and returns
			// the reassembled NAL on the final fragment.
			nalData, err := h264.Unmarshal(pkt.Payload)
			if err != nil || len(nalData) == 0 {
				continue
			}

			out.Write(nalData)

			// Flush on RTP marker — end of access unit (complete frame)
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

	// ── 3. Set remote description (camera's offer) ──────────────────────────
	if err := pc.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeOffer,
		SDP:  offer.SDP,
	}); err != nil {
		return fmt.Errorf("set remote description: %w", err)
	}

	// ── 4. Create answer and gather ICE candidates ──────────────────────────
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

	// ── 5. Send answer to camera ────────────────────────────────────────────
	ansPayload, _ := json.Marshal(map[string]string{
		"session_id": offer.SessionID,
		"sdp":        pc.LocalDescription().SDP,
	})
	ans, err := http.Post(baseURL+"/rpc/Streamer.Answer",
		"application/json", bytes.NewReader(ansPayload))
	if err != nil {
		return fmt.Errorf("send answer: %w", err)
	}
	ansBody, _ := io.ReadAll(ans.Body)
	ans.Body.Close()
	log.Printf("[webrtc] answer sent, camera replied: %s", ansBody)

	// ── 6. Wait until stream ends or ICE fails ──────────────────────────────
	return <-done
}
