# FlyDecoder

**Author:** [kristoferUA](https://github.com/kristoferUA)

A local Python and Pygame desktop app for decoding Base64, hexadecimal, binary, and URL-encoded strings. FlyDecoder sends numeric input features through the CPU FlyBrain simulator using the official MaleCNS v1.0 connectome.

## Features

- Decode Base64, hexadecimal, binary, and URL-encoded strings, or recognize plain text.
- Use the full FlyBrain connectome simulation on CPU; CUDA and API keys are not required.
- Train and save a small external readout locally between sessions.
- Watch an animated fly inspect the input and choose a decoder.

## Run on Windows

1. Run `install_flydecoder.bat` once. It installs Python 3.12 and the application dependencies, then downloads and verifies the three required MaleCNS files (about 1.1 GB total) and builds the local graph cache.
2. Start the app with `run.bat`. Later launches work offline.

The first setup needs about 8 GB of free RAM. The downloaded tables and graph cache stay on your computer and are not included in this repository.

## How it works

The app extracts 24 numeric features from the input without passing the expected encoding to the simulation. An external readout maps the simulated activity to a decoder choice. When needed, the app tries other decoders and scores whether they recover readable text. The readout is stored under `out/` and remains local.

## Credits and licenses

- FlyBrain simulator source: [HEREISCB/flybrain](https://github.com/HEREISCB/flybrain), © Chaitanya Benade and contributors, MIT. See [LICENSE](LICENSE).
- MaleCNS v1.0 connectome data: FlyEM at HHMI Janelia, Google Research, University of Cambridge, and MRC Laboratory of Molecular Biology; CC BY 4.0. See [MALECNS_ATTRIBUTION.md](MALECNS_ATTRIBUTION.md).

For the detailed guide in Ukrainian, see [README_UA.md](README_UA.md).