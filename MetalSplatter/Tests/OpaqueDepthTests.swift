import XCTest
import Metal
import simd
@testable import MetalSplatter

final class OpaqueDepthTests: XCTestCase {
    func testOcclusionAndOptionalInputInBothPipelines() async throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        for highQualityDepth in [false, true] {
            let renderer = try SplatRenderer(
                device: device, colorFormat: .rgba8Unorm,
                depthFormat: highQualityDepth ? .depth32Float : .invalid,
                sampleCount: 1, maxViewCount: 1, maxSimultaneousRenders: 1,
                highQualityDepth: highQualityDepth
            )
            let buffer = try MetalBuffer<EncodedSplatPoint>(device: device, capacity: 1)
            buffer.count = 1
            buffer.values[0] = EncodedSplatPoint(
                position: SIMD3(0, 0, -1), colorSH0: (SIMD3<Float>(1, 0, 0) - 0.5) / 0.28209479177387814,
                opacity: 0.9, scale: SIMD3(repeating: 0.2), rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
            )
            let sorted = DispatchSemaphore(value: 0)
            renderer.afterNextSort { sorted.signal() }
            await renderer.addChunk(SplatChunk(splats: buffer))
            XCTAssertEqual(sorted.wait(timeout: .now() + 5), .success)

            let color = try texture(device, format: .rgba8Unorm, usage: .renderTarget)
            let opaque = try texture(device, format: .r32Float, usage: .shaderRead)
            let outputDepth = highQualityDepth ? try texture(device, format: .depth32Float, usage: .renderTarget) : nil
            let viewport = SplatRenderer.ViewportDescriptor(
                viewport: MTLViewport(originX: 0, originY: 0, width: 32, height: 32, znear: 0, zfar: 1),
                projectionMatrix: simd_float4x4(columns: (
                    SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0),
                    SIMD4(0, 0, -10 / 9.9, -1), SIMD4(0, 0, -1 / 9.9, 0)
                )), viewMatrix: matrix_identity_float4x4, screenSize: SIMD2(32, 32)
            )
            for limit: Float? in [0.5, 1.0, nil] {
                if let limit {
                    let values = [Float](repeating: limit, count: 32 * 32)
                    values.withUnsafeBytes { bytes in
                        opaque.replace(region: MTLRegionMake2D(0, 0, 32, 32), mipmapLevel: 0,
                                       withBytes: bytes.baseAddress!, bytesPerRow: 32 * 4)
                    }
                }
                let commands = try XCTUnwrap(queue.makeCommandBuffer())
                let rendered = try renderer.render(
                    viewports: [viewport], colorTexture: color, colorStoreAction: .store,
                    depthTexture: outputDepth, rasterizationRateMap: nil, renderTargetArrayLength: 1,
                    sortTimeout: 5, opaqueDepthTexture: limit == nil ? nil : opaque, to: commands
                )
                XCTAssertTrue(rendered)
                await withCheckedContinuation { continuation in
                    commands.addCompletedHandler { _ in continuation.resume() }
                    commands.commit()
                }
                XCTAssertNil(commands.error)
                var pixels = [UInt8](repeating: 0, count: 32 * 32 * 4)
                pixels.withUnsafeMutableBytes { bytes in
                    color.getBytes(bytes.baseAddress!, bytesPerRow: 32 * 4,
                                   from: MTLRegionMake2D(0, 0, 32, 32), mipmapLevel: 0)
                }
                let centerPixel: Int = 16 * 32 + 16
                let alpha: UInt8 = pixels[centerPixel * 4 + 3]
                if limit == 0.5 {
                    XCTAssertEqual(alpha, 0)
                } else {
                    XCTAssertGreaterThan(alpha, 100)
                }
            }
        }
    }

    func testMismatchedOpaqueDepthFailsBeforeRendering() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let renderer = try SplatRenderer(device: device, colorFormat: .rgba8Unorm, depthFormat: .invalid,
                                         sampleCount: 1, maxViewCount: 1, maxSimultaneousRenders: 1)
        let color = try texture(device, format: .rgba8Unorm, usage: .renderTarget)
        let wrongFormat = try texture(device, format: .rgba8Unorm, usage: .shaderRead)
        let commands = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
        XCTAssertThrowsError(try renderer.render(
            viewports: [], colorTexture: color, colorStoreAction: .store, depthTexture: nil,
            rasterizationRateMap: nil, renderTargetArrayLength: 1,
            opaqueDepthTexture: wrongFormat, to: commands
        ))
    }

    private func texture(_ device: MTLDevice, format: MTLPixelFormat, usage: MTLTextureUsage) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: 32, height: 32, mipmapped: false)
        descriptor.storageMode = format == .depth32Float ? .private : .shared
        descriptor.usage = usage
        return try XCTUnwrap(device.makeTexture(descriptor: descriptor))
    }
}
