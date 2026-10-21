# Adding a language

The camera **Describe** button answers in the user's language. Every language
is one JSON file in [`app/locales/`](../app/locales/). Adding a language needs
no Python changes.

## How the language is chosen

1. `?lang=xx` on the request. The web UI sends the browser's language
   (`navigator.language`). To override it in the browser console, run:
   `localStorage.setItem("hubLang", "bg")`
2. The `Accept-Language` header, for API callers.
3. The `APP_LANG` env var (`.env` / `docker-compose.yml`), which defaults to `en`.
4. English.

Regional tags fall back to the base language, so `de-AT` uses `de.json`.
The response includes a `"lang"` field that shows which language was used.

## Steps

1. Copy `app/locales/en.json` to `app/locales/<code>.json`. Use an ISO 639-1
   code such as `de`, `fr` or `es`. A regional file such as `pt-br.json` also
   works and is preferred over `pt.json` for `pt-BR` browsers.
2. Translate:
   - `_meta.name`: the language's own name, such as `"Deutsch"`
   - `phrases.seen`: the sentence around the list. Keep `{items}`.
   - `phrases.nothing`: the text shown when nothing is detected
   - `phrases.separator`: the text between items, usually `", "`
   - `phrases.count`: the format of one item. Keep `{n}` and `{word}`.
     Reorder them if your language needs it.
   - `errors.*`: the error messages
   - `labels.0` … `labels.79`: the YOLO / COCO object names, with `one` for
     the singular and `other` for the plural. Don't change the numeric keys.
3. Rebuild with `docker compose up -d --build app`.
4. Check it:
   - `curl localhost/api/languages` should list the new language.
   - At startup the logs show `Locales loaded: ...`. A warning is logged if
     any keys are missing. Missing keys fall back to English.

To test without a camera, run this from the repo root:

```sh
python3 -c "from app import i18n; print(i18n.describe('de', [{'class_id': 0}, {'class_id': 0}, {'class_id': 2}]))"
```
