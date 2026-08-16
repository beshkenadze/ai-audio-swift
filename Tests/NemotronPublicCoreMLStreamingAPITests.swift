#if canImport(CoreML)
    import MLX
    import MLXAudioSTT
    import Testing

    @Suite("Nemotron Public CoreML Streaming API Tests")
    struct NemotronPublicCoreMLStreamingAPITests {
        @Test
        func `CoreML streaming encoder entry point is public`() {
            let entryPoint: (
                NemotronASRModel,
                MLXArray,
                NemotronCoreMLStreamingEncoder,
                (MLXArray) -> Void,
            )
                throws -> Void = { model, mel, encoder, onChunk in
                    try model.cacheAwareStreamEncodeCoreML(
                        mel,
                        language: nil,
                        encoder: encoder,
                        onChunk: onChunk,
                    )
                }

            _ = entryPoint
        }
    }
#endif
