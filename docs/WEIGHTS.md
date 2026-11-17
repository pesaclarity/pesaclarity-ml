# Plain weights format, version 1

pesaclarity-app runs its small models in Dart instead of a native runtime (pesaclarity/docs,
decision 0019). For each such model akili writes, beside the TFLite file, a `.weights` file:
the tensors TFLite runs, in a small format any language can read in a few dozen lines. The
TFLite file stays the reference: akili's tests and evaluation run it, and the weights are read
back out of it ([`akili.ask.reader_weights`](../src/akili/ask/reader_weights.py)), so they are
byte for byte the int8 and float32 values in it.

Today one model has one: the Ask reader (`ask-reader.weights`, about 249 KB).

## Layout

Little-endian throughout.

| Bytes | Field |
| --- | --- |
| 0-3 | magic, ASCII `PCWT` |
| 4-7 | uint32 format version, `1` |
| 8-11 | uint32 header length `H` in bytes |
| 12 to 12+H | the header: UTF-8 JSON, padded with spaces so that 12+H is a multiple of 4 |
| 12+H to the end | the tensor data |

The file ends exactly after the last tensor (and its padding). A reader refuses a file whose
magic, version or sizes do not fit, rather than guessing.

## Header

```json
{
  "format_version": 1,
  "model": "ask-reader",
  "source": {"file": "ask-reader.tflite", "sha256": "f665…", "size_bytes": 271904},
  "layers": [
    {"name": "token_embedding", "kind": "embedding"},
    {"name": "conv_3", "kind": "conv1d", "size": 3, "dilation": 3, "activation": "relu"},
    {"name": "intent", "kind": "dense", "activation": "softmax"}
  ],
  "tensors": [
    {"name": "conv_3.kernel", "dtype": "int8", "shape": [96, 1, 3, 96],
     "offset": 0, "bytes": 27648,
     "quantization": {"axis": 0, "zero_point": 0, "scales_offset": 27648, "scales_bytes": 384}},
    {"name": "conv_3.bias", "dtype": "float32", "shape": [96], "offset": 28032, "bytes": 384}
  ]
}
```

- `format_version`: the same number as bytes 4-7.
- `model`: which model this is; the reader of a file knows the network it runs.
- `source`: the TFLite file the weights were read from, with its SHA-256 and size, so the
  weights trace back to the file the release scored.
- `layers`: the layers in the order they run. `kind` is `embedding`, `conv1d` (with kernel
  `size` and `dilation`, "same" padding) or `dense`; `activation` is `relu` or `softmax`. A
  program that runs the model checks this list is the one it implements.
- `tensors`: every tensor, in any order. A layer `x` has `x.table` (an embedding's rows) or
  `x.kernel` and `x.bias`.
  - `dtype`: `int8` or `float32`.
  - `shape`: row-major (C order), as in the TFLite graph: a dense kernel is `[out, in]`, a
    1-D convolution's kernel is TFLite's `[out, 1, size, in]`, an embedding table
    `[rows, width]`.
  - `offset`, `bytes`: where its values are, from the start of the tensor data. Every offset
    is a multiple of 4; `bytes` is the element count times the element size, and the next
    offset is `bytes` rounded up to 4.
  - `quantization` (int8 only): the real value of an element is
    `scales[c] * (value - zero_point)`, with `scales` the float32 array at `scales_offset`
    (`scales_bytes` long) and `c` the element's index along dimension `axis`; when `axis` is
    `null` there is one scale for the whole tensor. TFLite's weight quantisation is symmetric,
    so `zero_point` is 0 today.

Unknown header fields are ignored, so fields can be added within a version. Anything that
changes how existing fields are read (a new dtype, layout or layer kind a reader must know)
bumps `format_version`, and the app refuses a version it does not read.

## The rayo graph

The app's native runtime, [rayo](https://github.com/pesaclarity/rayo), runs the same file. For
it the header has an optional `graph` field (and may have a `metadata` object): the ops that run
the tensors, with their inputs and attributes, specified in rayo's
[docs/FORMAT.md](https://github.com/pesaclarity/rayo/blob/main/docs/FORMAT.md). A reader that
does not know the field ignores it, as above, so adding it keeps the file at version 1, and the
Dart reader and rayo read one file.

```json
"graph": {
  "version": 1,
  "symbols": {"T": 96},
  "inputs": [{"name": "tokens", "dtype": "int32", "shape": ["T"]},
             {"name": "shapes", "dtype": "int32", "shape": ["T"]}],
  "nodes": [
    {"op": "gather", "inputs": ["token_embedding.table", "tokens"], "output": "tok"},
    {"op": "conv1d", "inputs": ["emb", "conv_1.kernel", "conv_1.bias"], "output": "c1",
     "size": 3, "dilation": 1, "activation": "relu", "input_quantization": "tensor_phases"}
  ],
  "outputs": [{"name": "slots", "node": "slots"}, {"name": "intent", "node": "intent"}]
}
```

`input_quantization` names how TFLite's hybrid kernels quantise the layer's input, as described
below: `rows` (each row) or `tensor_phases` (the whole input, each dilation phase alone). The
Ask reader's graph is [`rayo_graph`](../src/akili/ask/reader_weights.py); `graph.version` is
rayo's, separate from this file's version.

## The manifest

The model's manifest carries the file beside the TFLite fields, which stay for provenance:

```json
"weights_file": "ask-reader.weights",
"weights_sha256": "…",
"weights_size_bytes": 248684,
"weights_format_version": 1
```

The app checks the size and SHA-256 before it reads the weights, and uses its rules instead of
the model when they do not match or the version is one it does not read.

## The Ask reader on these weights

Its network is [`build_ask_reader`](../src/akili/ask/reader.py): embeddings, three residual
convolutions (the third dilated), a slot head (dense, two residual convolutions, dense) and an
intent head on masked max and mean pooling. The model is dynamic-range quantised: int8 weights,
float32 everything else. To give TFLite's outputs, a program must also run it the way TFLite's
hybrid kernels do: before each int8 layer, quantise its float input to int8 on the fly,
asymmetrically, one scale and zero point per input row (each token's row for a dense layer, the
whole question for a convolution), the range widened to include 0, rounding halves away from
zero; a dilated convolution runs as `SPACE_TO_BATCH_ND`, so each of its `dilation` phases (the
rows `i` with `i % dilation == p`) is a row of its own. [`forward`](../src/akili/ask/reader_weights.py)
is the reference, in NumPy; the app's Dart reader copies it.

Done that way, the outputs are TFLite's to about 1e-4 on the released reader. TFLite's own
kernels disagree with each other at about that level (XNNPACK and the built-in kernels can
round one int8 value differently), so the labels are the check, and the probabilities are
close, not bit-exact.

## Exporting

`akili ask-reader` writes the weights and their manifest fields with every export. For a
TFLite file exported before, `akili ask-reader-weights --model path/to/ask-reader.tflite`
writes `ask-reader.weights` beside it (or into `--out`) and, when the manifest there is that
model's, adds the fields to it. The same TFLite file always gives the same bytes.
