import Metal
import simd
import Synchronization
import XCTest
@testable import MetalSplatter

private final class Counter: Sendable {
    private let value = Mutex(0)
    var count: Int { value.withLock { $0 } }
    func increment() { value.withLock { $0 += 1 } }
}

final class CurrentPoseSortTests: XCTestCase {
    func chunk(_ device: MTLDevice, positions: [SIMD3<Float>], index: UInt16 = 0) throws -> SplatSorter.ChunkReference {
        let buffer = try MetalBuffer<EncodedSplatPoint>(device: device, capacity: positions.count)
        for position in positions {
            buffer.append(EncodedSplatPoint(
                position: position, colorSH0: .zero, opacity: 1, scale: SIMD3(repeating: 0.1),
                rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
            ))
        }
        return SplatSorter.ChunkReference(chunkIndex: index, buffer: buffer)
    }

    func indices(_ sorter: SplatSorter, pose: SplatSorter.CameraPose) async throws -> [ChunkedSplatIndex] {
        let deadline = Date().addingTimeInterval(2)
        while Date() < deadline {
            if let buffer = sorter.tryObtainSortedIndices(for: pose) {
                defer { sorter.releaseSortedIndices(buffer) }
                return (0..<buffer.count).map { buffer.values[$0] }
            }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        throw NSError(domain: "CurrentPoseSortTests", code: 1)
    }

    func testRepeatedPoseReusesSort() async throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let sorter = try SplatSorter(device: device)
        let pose = SplatSorter.CameraPose(position: .zero, forward: SIMD3(0, 0, -1))
        let counter = Counter()
        sorter.onSortStart = { counter.increment() }
        sorter.setChunks([try chunk(device, positions: [SIMD3(-1, 0, 0), SIMD3(2, 0, 0)])])
        let first = try await indices(sorter, pose: pose)
        XCTAssertEqual(first.map(\.splatIndex), [1, 0])
        for _ in 0..<20 {
            sorter.updateCameraPose(position: pose.position, forward: pose.forward)
            _ = try await indices(sorter, pose: pose)
        }
        try await Task.sleep(nanoseconds: 20_000_000)
        XCTAssertEqual(counter.count, 1)
    }

    func testChangedPoseCannotReadPreviousSort() async throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let sorter = try SplatSorter(device: device)
        let a = SplatSorter.CameraPose(position: SIMD3(-3, 0, 0), forward: SIMD3(0, 0, -1))
        let b = SplatSorter.CameraPose(position: SIMD3(3, 0, 0), forward: a.forward)
        sorter.updateCameraPose(position: a.position, forward: a.forward)
        sorter.setChunks([try chunk(device, positions: [SIMD3(-1, 0, 0), SIMD3(1, 0, 0)])])
        let first = try await indices(sorter, pose: a)
        XCTAssertEqual(first.map(\.splatIndex), [1, 0])
        let started = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        sorter.onSortStart = { started.signal(); _ = release.wait(timeout: .now() + 2) }
        defer { release.signal() }
        sorter.updateCameraPose(position: b.position, forward: b.forward)
        XCTAssertEqual(started.wait(timeout: .now() + 2), .success)
        XCTAssertNil(sorter.tryObtainSortedIndices(for: b))
        let legacy = try XCTUnwrap(sorter.tryObtainSortedIndices())
        sorter.releaseSortedIndices(legacy)
        release.signal()
        let next = try await indices(sorter, pose: b)
        XCTAssertEqual(next.map(\.splatIndex), [0, 1])
    }

    func testSamePoseAfterChunkChangeRequiresNewSort() async throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let sorter = try SplatSorter(device: device)
        let pose = SplatSorter.CameraPose(position: .zero, forward: SIMD3(0, 0, -1))
        sorter.setChunks([try chunk(device, positions: [SIMD3(1, 0, 0)])])
        _ = try await indices(sorter, pose: pose)
        let started = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        sorter.onSortStart = { started.signal(); _ = release.wait(timeout: .now() + 2) }
        defer { release.signal() }
        try await sorter.withExclusiveAccess(invalidateIndexBuffers: false) {
            try sorter.addChunkToSort(chunk(device, positions: [SIMD3(4, 0, 0)], index: 1))
        }
        XCTAssertEqual(started.wait(timeout: .now() + 2), .success)
        XCTAssertNil(sorter.tryObtainSortedIndices(for: pose))
        release.signal()
        let next = try await indices(sorter, pose: pose)
        XCTAssertEqual(next.map(\.chunkIndex), [1, 0])
    }

    func testPoseChangeDuringSortCannotPublishAsCurrent() async throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let sorter = try SplatSorter(device: device)
        let a = SplatSorter.CameraPose(position: .zero, forward: SIMD3(0, 0, -1))
        let b = SplatSorter.CameraPose(position: SIMD3(3, 0, 0), forward: a.forward)
        let started = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        sorter.onSortStart = { started.signal(); _ = release.wait(timeout: .now() + 2) }
        defer { release.signal() }
        sorter.setChunks([try chunk(device, positions: [SIMD3(1, 0, 0), SIMD3(2, 0, 0)])])
        XCTAssertEqual(started.wait(timeout: .now() + 2), .success)
        sorter.updateCameraPose(position: b.position, forward: b.forward)
        release.signal()
        XCTAssertEqual(started.wait(timeout: .now() + 2), .success)
        XCTAssertNil(sorter.tryObtainSortedIndices(for: b))
        let previous = try XCTUnwrap(sorter.tryObtainSortedIndices(for: a))
        sorter.releaseSortedIndices(previous)
        release.signal()
        let next = try await indices(sorter, pose: b)
        XCTAssertEqual(next.map(\.splatIndex), [0, 1])
    }
}
