# Nakdimon web demo

Build (compiles `packages/nakdimon-js` if needed, copies `lib/` and the model into `web/`):

    ./build.sh

Serve locally:

    python3 -m http.server --directory web

Then open http://localhost:8000/. `lib/` and `Nakdimon.onnx` are build products, copied by `build.sh` and not committed.
