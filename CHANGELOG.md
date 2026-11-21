# Changelog

## [0.3.0](https://github.com/pesaclarity/pesaclarity-ml/compare/v0.2.0...v0.3.0) (2026-11-21)


### ⚠ BREAKING CHANGES

* the --keras flags, the ask-writer, ask-writer-weights and ask-reader-weights commands, the [train] extra and export.quantize are gone.

### Features

* ask reader v3 reads the version 2 misses ([#51](https://github.com/pesaclarity/pesaclarity-ml/issues/51)) ([d811635](https://github.com/pesaclarity/pesaclarity-ml/commit/d811635eeb2c927c7660e8fede6f99bbc70af5cd))
* **ask:** the Ask reader in PyTorch, shipped as rayo int8 weights ([6a44476](https://github.com/pesaclarity/pesaclarity-ml/commit/6a444764dd581fcadfe8dbbe4c7e139c66edee66))
* **ask:** version 4 templates for money in from short names, payees in compare and trend, and amounts typed many ways ([839ad4d](https://github.com/pesaclarity/pesaclarity-ml/commit/839ad4d5e0acc309f701289390bfdd2f9f31eddf))
* **categoriser:** the categoriser in PyTorch, shipped as rayo int8 weights ([064a453](https://github.com/pesaclarity/pesaclarity-ml/commit/064a4532c164f4f184ed7855d2315799a94bbf8a))
* export the ask reader's plain weights for the app's Dart reader ([#46](https://github.com/pesaclarity/pesaclarity-ml/issues/46)) ([e45e30d](https://github.com/pesaclarity/pesaclarity-ml/commit/e45e30d0e8a4a552a4a3cdb941a0f277e8fc5e21))
* export the ask writer's plain weights, with the graph rayo decodes it with ([#48](https://github.com/pesaclarity/pesaclarity-ml/issues/48)) ([1732297](https://github.com/pesaclarity/pesaclarity-ml/commit/1732297d968d98497510fe7e00837c9d9e051891))
* **privacy:** Opacus's accountants in place of dp-accounting ([080f25b](https://github.com/pesaclarity/pesaclarity-ml/commit/080f25b0a1087d3acd929e07ec7b1bde8d031b51))
* **reader:** the message reader in PyTorch, shipped as rayo's int8 reader.weights ([f830212](https://github.com/pesaclarity/pesaclarity-ml/commit/f830212b6fd8ad39ed5496fd99c2a6f3062f6447))
* remove TensorFlow, Keras, TFLite and dp-accounting ([6d1f75b](https://github.com/pesaclarity/pesaclarity-ml/commit/6d1f75b56486980dcf70c1af0713a9677810fdd1))
* the ask models' rayo graphs from rayo-convert, not wired by hand ([#50](https://github.com/pesaclarity/pesaclarity-ml/issues/50)) ([630105b](https://github.com/pesaclarity/pesaclarity-ml/commit/630105b609b5845a80cdda77109a3ff171a906e5))
* the ask reader's rayo graph in rayo's general ops ([#49](https://github.com/pesaclarity/pesaclarity-ml/issues/49)) ([991a3c2](https://github.com/pesaclarity/pesaclarity-ml/commit/991a3c2a5d20c5baf7f259b47fbca9a8ac678c8b))
* the ask reader's weights carry the graph rayo runs them with ([#47](https://github.com/pesaclarity/pesaclarity-ml/issues/47)) ([48e53fe](https://github.com/pesaclarity/pesaclarity-ml/commit/48e53fecdd92debd9defe92871e8d55a21f57513))
* the torch export path, a torch module to rayo weights via ONNX ([97174b2](https://github.com/pesaclarity/pesaclarity-ml/commit/97174b23993e98725c4496d016de384c1e9c84d7))


### Fixes

* **ask:** count-number and repaid-loan look-alikes, no bare-digit amounts, version 4 at half weight, 30 epochs ([1635bff](https://github.com/pesaclarity/pesaclarity-ml/commit/1635bffd7937599005ecffc85a0aefc9bf097c02))
* **ask:** draw version 4 templates as often as version 1's, keep plain words out of one-word names and years out of bare amounts ([eaf157d](https://github.com/pesaclarity/pesaclarity-ml/commit/eaf157d0532f970107c916f459013efbe1792b3e))
* **ask:** first names alone only in money-out templates ([10323bf](https://github.com/pesaclarity/pesaclarity-ml/commit/10323bf5273d6dcda2a1d61b893893335d6c3912))
* **ask:** no evaluation Sacco names in training, AdamW weight decay, first names in templates of their own ([1785af4](https://github.com/pesaclarity/pesaclarity-ml/commit/1785af4333f7109073495fdea0555958b273eab4))
* **ask:** the reader's reference forward pass runs in float32 in rayo's order ([cba3daf](https://github.com/pesaclarity/pesaclarity-ml/commit/cba3daf22820e3f63bbb35167409daa6e8a4cc15))
* **check:** fetch rayo's sources outside the hook's git environment ([ca0a5f0](https://github.com/pesaclarity/pesaclarity-ml/commit/ca0a5f06a87e17bbc5374fae61196ae789c27682))
* **export:** keep layer names in the ONNX export ([9cba103](https://github.com/pesaclarity/pesaclarity-ml/commit/9cba103611b523af3d4392735643ecf92bbb78ec))

## [0.2.0](https://github.com/pesaclarity/pesaclarity-ml/compare/v0.1.0...v0.2.0) (2026-11-17)


### Features

* Ask reader and writer models ([#40](https://github.com/pesaclarity/pesaclarity-ml/issues/40)) ([dfb9c24](https://github.com/pesaclarity/pesaclarity-ml/commit/dfb9c2493cc230bc1e3c90ad7e02896724dd5ed8))
* ask reader that tags names and money in ([#43](https://github.com/pesaclarity/pesaclarity-ml/issues/43)) ([d51d5bc](https://github.com/pesaclarity/pesaclarity-ml/commit/d51d5bc6a6a6e5769210b5625c42566b52c5a05c))
* signed-off privacy limits, with the PLD epsilon reported ([#31](https://github.com/pesaclarity/pesaclarity-ml/issues/31)) ([760b496](https://github.com/pesaclarity/pesaclarity-ml/commit/760b4966f96676655d10c044da9e3f179282b91d))
