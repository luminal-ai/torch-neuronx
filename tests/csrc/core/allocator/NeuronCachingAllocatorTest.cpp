#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstring>
#include <algorithm>
#include <thread>
#include <chrono>
#include <atomic>
#include <map>
#include <unordered_map>

#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockNeuronDevice.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/allocator/NeuronSegmentAllocator.h"
#include "torch_neuronx/csrc/core/allocator/ShelvesCleanup.h"
#include "torch_neuronx/csrc/core/utils/CopyUtils.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMemorySnapshot.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

using namespace c10_neuron;
using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::AnyNumber;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::SetArgPointee;

class NeuronCachingAllocatorTest : public ::testing::Test {
 protected:
  static bool streams_initialized_;

  void SetUp() override {
    mock_session_ = std::make_unique<MockNRTSession>();
    next_addr_ = 0x10000;

    // Mock device count: 1 fake Neuron device for testing
    c10_neuron::testing::SetMockDeviceCount(1);

    // Mock nrt_get_visible_vnc_count so stream pool init discovers 1 device
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_get_visible_vnc_count(_))
        .Times(AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(1u), Return(NRT_SUCCESS)));

    // Initialize stream pools once (needed for NeuronEvent::record in neuron_deleter)
    if (!streams_initialized_) {
      at::neuron::InitializeStreamPools();
      streams_initialized_ = true;
    }

    // Mock NRT tensor allocation: return fake tensor pointers with unique addresses
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke(
            [this](nrt_tensor_placement_t, int32_t, size_t size, const char*, nrt_tensor_t** out) {
              auto* fake = reinterpret_cast<nrt_tensor_t*>(next_addr_);
              next_addr_ += std::max(size, static_cast<size_t>(0x1000));
              *out = fake;
              return NRT_SUCCESS;
            }));

    // nrt_tensor_get_va: return address of the tensor itself (identity mapping)
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](const nrt_tensor_t* t) {
          return const_cast<void*>(reinterpret_cast<const void*>(t));
        }));

    // nrt_tensor_allocate_slice: return base + offset as fake slice handle
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate_slice(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke(
            [](const nrt_tensor_t* base, size_t offset, size_t, const char*, nrt_tensor_t** out) {
              auto base_addr = reinterpret_cast<uintptr_t>(base);
              *out = reinterpret_cast<nrt_tensor_t*>(base_addr + offset);
              return NRT_SUCCESS;
            }));

    // nrt_tensor_free: just null out the pointer
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](nrt_tensor_t** ptr) {
          if (ptr) *ptr = nullptr;
        }));

    // Mock async NRT operations so the worker thread can process kAlloc/kDealloc.
    // nrta_event_register_seq_id_completion: signal the eventfd immediately
    // so the runtime worker wakes from read(efd) and processes completions.
    EXPECT_CALL(*MockNRT::GetInstance(), nrta_event_register_seq_id_completion(_, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](nrta_seq_t, int fd) -> NRT_STATUS {
          uint64_t val = 1;
          ::write(fd, &val, sizeof(val));
          return NRT_SUCCESS;
        }));

    // nrta_get_sequence: return UINT64_MAX to indicate all operations completed.
    EXPECT_CALL(*MockNRT::GetInstance(), nrta_get_sequence(_, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<3>(UINT64_MAX), Return(NRT_SUCCESS)));

    // nrta_tensor_write: the scalar cache miss path submits a kWrite for the
    // value. Set the async-status out-param (arg 6) to NRT_SUCCESS so the op's
    // completion status isn't the fail-closed NRT_FAILURE default, and a nonzero
    // sequence out-param (arg 7) to match how the other async NRT ops are mocked.
    EXPECT_CALL(*MockNRT::GetInstance(), nrta_tensor_write(_, _, _, _, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(
            DoAll(SetArgPointee<6>(NRT_SUCCESS), SetArgPointee<7>(1u), Return(NRT_SUCCESS)));

    // Initialize the execution engine (needed for lazy alloc SubmitOperationContext)
    at::neuron::NeuronResourceManager::Instance();
  }

  void TearDown() override {
    // Synchronize all devices to complete pending async operations (lazy alloc kAlloc/kDealloc)
    for (int device = 0; device < c10_neuron::testing::device_count(); ++device) {
      at::neuron::synchronize(device);
    }
    // Drain segment pools back to NRT so state doesn't leak across tests.
    NeuronCachingAllocator::emptyCache();
    mock_session_.reset();
  }

  /// allocate and return the DataPtr.
  c10::DataPtr allocate(size_t size) {
    auto* allocator = NeuronCachingAllocator::get();
    return allocator->allocate(size);
  }

  /// Synchronize device 0 to wait for pending kAlloc/kDealloc to complete.
  /// Call this after allocate() for findTensors to work.
  void syncDevice() { at::neuron::synchronize(0); }

  // Get a valid non-default c10::Stream for device 0.
  static c10::Stream getNonDefaultStream() {
    return c10::Stream(c10::Stream::UNSAFE, c10::Device(c10::DeviceType::PrivateUse1, 0),
                       /*stream_id=*/32);
  }

  std::unique_ptr<MockNRTSession> mock_session_;
  uintptr_t next_addr_ = 0x10000;
};

bool NeuronCachingAllocatorTest::streams_initialized_ = false;

TEST_F(NeuronCachingAllocatorTest, AllocateReturnsValidDataPtr) {
  auto data_ptr = allocate(1024);

  EXPECT_NE(data_ptr.get(), nullptr);
  EXPECT_EQ(data_ptr.device().type(), c10::DeviceType::PrivateUse1);
  EXPECT_EQ(data_ptr.device().index(), 0);
}

TEST_F(NeuronCachingAllocatorTest, AllocateZeroReturnsNull) {
  auto data_ptr = allocate(0);

  EXPECT_EQ(data_ptr.get(), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, MultipleAllocationsReturnDistinctPointers) {
  auto dp1 = allocate(1024);
  auto dp2 = allocate(2048);
  auto dp3 = allocate(512);

  EXPECT_NE(dp1.get(), nullptr);
  EXPECT_NE(dp2.get(), nullptr);
  EXPECT_NE(dp3.get(), nullptr);

  // All pointers must be distinct
  EXPECT_NE(dp1.get(), dp2.get());
  EXPECT_NE(dp1.get(), dp3.get());
  EXPECT_NE(dp2.get(), dp3.get());
}

TEST_F(NeuronCachingAllocatorTest, FindTensorReturnsNullForNullptr) {
  EXPECT_EQ(NeuronCachingAllocator::findTensor(nullptr), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, FindTensorReturnsNullForUnknownPtr) {
  void* bogus = reinterpret_cast<void*>(0xDEADBEEF);
  EXPECT_EQ(NeuronCachingAllocator::findTensor(bogus), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, FindTensorReturnsHandleForAllocatedPtr) {
  auto dp = allocate(1024);
  void* ptr = dp.get();
  ASSERT_NE(ptr, nullptr);

  syncDevice();

  nrt_tensor_t* handle = NeuronCachingAllocator::findTensor(ptr);
  EXPECT_NE(handle, nullptr);
}

TEST_F(NeuronCachingAllocatorTest, FindTensorPtrReturnsSharedPtrForAllocatedPtr) {
  auto dp = allocate(1024);
  void* ptr = dp.get();
  ASSERT_NE(ptr, nullptr);

  syncDevice();

  auto tensor_ptr = NeuronCachingAllocator::findTensorPtr(ptr);
  EXPECT_NE(tensor_ptr, nullptr);

  // The shared_ptr's deleter frees the NRT slice — should not crash
  tensor_ptr.reset();
}

TEST_F(NeuronCachingAllocatorTest, FindTensorPtrReturnsNullForUnknownPtr) {
  void* bogus = reinterpret_cast<void*>(0xDEADBEEF);
  EXPECT_EQ(NeuronCachingAllocator::findTensorPtr(bogus), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, GetTensorFromContextThrowsOnNull) {
  EXPECT_THROW(NeuronCachingAllocator::getTensorFromContext(nullptr), c10::Error);
}

TEST_F(NeuronCachingAllocatorTest, GetTensorFromContextReturnsHandleForAllocatedPtr) {
  auto dp = allocate(1024);
  void* ctx = dp.get_context();
  ASSERT_NE(ctx, nullptr);

  syncDevice();

  nrt_tensor_t* handle = NeuronCachingAllocator::getTensorFromContext(ctx);
  EXPECT_NE(handle, nullptr);

  // Should match findTensor for the same allocation
  nrt_tensor_t* find_handle = NeuronCachingAllocator::findTensor(dp.get());
  EXPECT_EQ(handle, find_handle);
}

TEST_F(NeuronCachingAllocatorTest, DeleterRemovesFromRegistryOnDestruction) {
  void* ptr = nullptr;
  {
    auto dp = allocate(1024);
    ptr = dp.get();
    ASSERT_NE(ptr, nullptr);

    syncDevice();

    // Should be findable while alive
    EXPECT_NE(NeuronCachingAllocator::findTensor(ptr), nullptr);
  }
  // DataPtr destructed, neuron_deleter called, registry entry removed
  // For lazy alloc, the deleter submits kDealloc; sync to let it complete
  syncDevice();

  EXPECT_EQ(NeuronCachingAllocator::findTensor(ptr), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, RecordStreamOnNullPtrDoesNotCrash) {
  c10::DataPtr empty_dp(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0));
  auto stream = getNonDefaultStream();
  EXPECT_NO_THROW(NeuronCachingAllocator::recordStream(empty_dp, stream));
}

TEST_F(NeuronCachingAllocatorTest, RecordStreamOnAllocatedPtrDoesNotCrash) {
  auto dp = allocate(1024);
  auto stream = getNonDefaultStream();
  EXPECT_NO_THROW(NeuronCachingAllocator::recordStream(dp, stream));
}

TEST_F(NeuronCachingAllocatorTest, SnapshotEmptyWhenNoActiveAllocations) {
  auto segments = NeuronCachingAllocator::getSnapshotSegments();
  EXPECT_TRUE(segments.empty());
}

TEST_F(NeuronCachingAllocatorTest, SnapshotReflectsActiveAllocations) {
  auto dp1 = allocate(1024);
  auto dp2 = allocate(2048);

  syncDevice();

  auto segments = NeuronCachingAllocator::getSnapshotSegments();
  // Should have at least one segment with active blocks
  EXPECT_GE(segments.size(), 1u);

  // Total active allocation size across all segments should account for dp1 + dp2
  size_t total_allocated = 0;
  for (const auto& seg : segments) {
    total_allocated += seg.allocated_size;
  }
  // Should be at least 1024 + 2048 = 3072 (may be rounded up by allocator)
  EXPECT_GE(total_allocated, 3072u);
}

TEST_F(NeuronCachingAllocatorTest, GetSegmentPoolReturnsSameInstanceForSameDevice) {
  auto& pool1 = c10_neuron::GetSegmentPool(0);
  auto& pool2 = c10_neuron::GetSegmentPool(0);
  EXPECT_EQ(&pool1, &pool2);
}

TEST_F(NeuronCachingAllocatorTest, EmptyCacheDoesNotCrash) {
  EXPECT_NO_THROW(NeuronCachingAllocator::emptyCache());
}

TEST_F(NeuronCachingAllocatorTest, RemoveUnknownPtrDoesNotCrash) {
  void* bogus = reinterpret_cast<void*>(0xDEADBEEF);
  EXPECT_NO_THROW(NeuronCachingAllocator::remove(bogus));
}

TEST_F(NeuronCachingAllocatorTest, FullLifecycleAllocFindRecordDestroy) {
  void* ptr = nullptr;
  nrt_tensor_t* handle = nullptr;
  {
    auto dp = allocate(2048);
    ptr = dp.get();
    ASSERT_NE(ptr, nullptr);

    syncDevice();

    // findTensor returns valid handle
    handle = NeuronCachingAllocator::findTensor(ptr);
    EXPECT_NE(handle, nullptr);

    // getTensorFromContext returns same handle
    void* ctx = dp.get_context();
    EXPECT_EQ(NeuronCachingAllocator::getTensorFromContext(ctx), handle);

    // recordStream on a non-default stream
    auto other_stream = getNonDefaultStream();
    NeuronCachingAllocator::recordStream(dp, other_stream);

    // Snapshot reflects the allocation
    auto segs = NeuronCachingAllocator::getSnapshotSegments();
    EXPECT_GE(segs.size(), 1u);
  }
  // After destruction, sync to let kDealloc complete
  syncDevice();

  // findTensor should return nullptr (removed from registry)
  EXPECT_EQ(NeuronCachingAllocator::findTensor(ptr), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, CachedScalarTensorIsResolvableByFindTensor) {
  int32_t val = 7;
  auto t =
      NeuronCachingAllocator::createCachedScalarTensor(&val, sizeof(val), c10::ScalarType::Int, 0);
  ASSERT_TRUE(t.has_value());
  void* ptr = t->data_ptr();
  ASSERT_NE(ptr, nullptr);

  syncDevice();

  nrt_tensor_t* handle = NeuronCachingAllocator::findTensor(ptr);
  EXPECT_NE(handle, nullptr);

  // The storage context resolves to the same handle.
  void* ctx = t->storage().data_ptr().get_context();
  EXPECT_EQ(NeuronCachingAllocator::getTensorFromContext(ctx), handle);
}

TEST_F(NeuronCachingAllocatorTest, IdenticalScalarsShareVAAndSlice) {
  int32_t val = 123;
  auto a =
      NeuronCachingAllocator::createCachedScalarTensor(&val, sizeof(val), c10::ScalarType::Int, 0);
  auto b =
      NeuronCachingAllocator::createCachedScalarTensor(&val, sizeof(val), c10::ScalarType::Int, 0);
  ASSERT_TRUE(a.has_value() && b.has_value());

  syncDevice();

  // Identical bit patterns dedup to one shared placeholder VA (the second call
  // is a cache hit that submits no new op) and resolve to the same slice.
  EXPECT_EQ(a->data_ptr(), b->data_ptr());
  EXPECT_EQ(NeuronCachingAllocator::findTensor(a->data_ptr()),
            NeuronCachingAllocator::findTensor(b->data_ptr()));
}

TEST_F(NeuronCachingAllocatorTest, ScalarEntrySurvivesTensorDestructionUntilEmptyCache) {
  int32_t val = 55;
  void* ptr = nullptr;
  {
    auto t = NeuronCachingAllocator::createCachedScalarTensor(&val, sizeof(val),
                                                              c10::ScalarType::Int, 0);
    ASSERT_TRUE(t.has_value());
    ptr = t->data_ptr();
    syncDevice();
    EXPECT_NE(NeuronCachingAllocator::findTensor(ptr), nullptr);
  }
  // The scalar entry is permanent (no-op deleter, shared across tensors), so
  // destroying the tensor must not free it.
  syncDevice();
  EXPECT_NE(NeuronCachingAllocator::findTensor(ptr), nullptr);

  // Only emptyCache tears it down.
  NeuronCachingAllocator::emptyCache();
  EXPECT_EQ(NeuronCachingAllocator::findTensor(ptr), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, DistinctScalarsResolveToDistinctSlices) {
  int32_t v1 = 1, v2 = 2;
  auto a =
      NeuronCachingAllocator::createCachedScalarTensor(&v1, sizeof(v1), c10::ScalarType::Int, 0);
  auto b =
      NeuronCachingAllocator::createCachedScalarTensor(&v2, sizeof(v2), c10::ScalarType::Int, 0);
  ASSERT_TRUE(a.has_value() && b.has_value());

  syncDevice();

  EXPECT_NE(a->data_ptr(), b->data_ptr());
  EXPECT_NE(NeuronCachingAllocator::findTensor(a->data_ptr()),
            NeuronCachingAllocator::findTensor(b->data_ptr()));
}

TEST_F(NeuronCachingAllocatorTest, OversizedScalarReturnsNullopt) {
  // Larger than kSlotSize (16 bytes) — not a real scalar; must fall back.
  char buf[32] = {};
  auto t =
      NeuronCachingAllocator::createCachedScalarTensor(buf, sizeof(buf), c10::ScalarType::Float, 0);
  EXPECT_FALSE(t.has_value());
}

// ---------------------------------------------------------------------------
// Device addresses filled on the stream worker (findPhysicalExtent,
// AddressWordsWriteKernelExecution).
// ---------------------------------------------------------------------------

// A fake NRT in which every handle is its VA, a segment's physical address is its VA plus
// kPhysicalShift, and a slice reports its segment's physical start and size (as NRT does).
class AddressWordsTest : public NeuronCachingAllocatorTest {
 protected:
  static constexpr uint64_t kPhysicalShift = 0x40'0000'0000ULL;

  void SetUp() override {
    NeuronCachingAllocatorTest::SetUp();
    auto* nrt = MockNRT::GetInstance();
    EXPECT_CALL(*nrt, nrt_tensor_allocate(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke(
            [this](nrt_tensor_placement_t, int32_t, size_t size, const char*, nrt_tensor_t** out) {
              uintptr_t va = next_addr_;
              next_addr_ += std::max(size, static_cast<size_t>(0x1000));
              std::lock_guard<std::mutex> lock(mutex_);
              segments_[va] = size;
              sizes_[va] = size;
              *out = reinterpret_cast<nrt_tensor_t*>(va);
              return NRT_SUCCESS;
            }));
    EXPECT_CALL(*nrt, nrt_tensor_allocate_slice(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([this](const nrt_tensor_t* base, size_t offset, size_t size,
                                      const char*, nrt_tensor_t** out) {
          uintptr_t va = reinterpret_cast<uintptr_t>(base) + offset;
          std::lock_guard<std::mutex> lock(mutex_);
          sizes_[va] = size;
          *out = reinterpret_cast<nrt_tensor_t*>(va);
          return NRT_SUCCESS;
        }));
    EXPECT_CALL(*nrt, nrt_tensor_get_size(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([this](const nrt_tensor_t* t) -> size_t {
          std::lock_guard<std::mutex> lock(mutex_);
          auto it = sizes_.find(reinterpret_cast<uintptr_t>(t));
          return it == sizes_.end() ? 0 : it->second;
        }));
    EXPECT_CALL(*nrt, nrt_tensor_get_device_allocation_info(_, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([this](const nrt_tensor_t* t,
                                      nrt_tensor_device_allocation_info_t* info) -> NRT_STATUS {
          uintptr_t va = reinterpret_cast<uintptr_t>(t);
          std::lock_guard<std::mutex> lock(mutex_);
          auto it = segments_.upper_bound(va);
          if (it == segments_.begin()) return NRT_FAILURE;
          --it;
          if (va >= it->first + it->second) return NRT_FAILURE;
          info->physical_address = it->first + kPhysicalShift;
          info->size = it->second;
          info->hbm_index = 0;
          return NRT_SUCCESS;
        }));
    EXPECT_CALL(*nrt, nrta_tensor_write(_, _, _, _, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([this](nrt_tensor_t* tensor, const void* buf, uint64_t offset,
                                      uint64_t size, int, int, NRT_STATUS* ret, nrta_seq_t* seq) {
          std::lock_guard<std::mutex> lock(mutex_);
          writes_.push_back({tensor, offset,
                             std::vector<uint8_t>(static_cast<const uint8_t*>(buf),
                                                  static_cast<const uint8_t*>(buf) + size)});
          *ret = NRT_SUCCESS;
          *seq = 1;
          return NRT_SUCCESS;
        }));
  }

  // The physical address the device reads for the storage at `ptr`, `byte_offset` bytes in:
  // its segment's physical start plus its offset in the segment.
  uint64_t ExpectedAddress(void* ptr, uint64_t byte_offset) {
    uintptr_t va = reinterpret_cast<uintptr_t>(NeuronCachingAllocator::findTensor(ptr));
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = --segments_.upper_bound(va);
    return it->first + kPhysicalShift + (va - it->first) + byte_offset;
  }

  struct Write {
    nrt_tensor_t* tensor;
    uint64_t offset;
    std::vector<uint8_t> bytes;
  };
  std::mutex mutex_;
  std::map<uintptr_t, size_t> segments_;
  std::unordered_map<uintptr_t, size_t> sizes_;
  std::vector<Write> writes_;
};

TEST_F(AddressWordsTest, PhysicalExtentIsTheSegmentStartPlusTheSliceOffset) {
  auto first = allocate(4096);
  auto second = allocate(8192);
  syncDevice();
  for (void* ptr : {first.get(), second.get()}) {
    auto extent = NeuronCachingAllocator::findPhysicalExtent(ptr);
    EXPECT_EQ(extent.address, ExpectedAddress(ptr, 0));
    EXPECT_GE(extent.bytes, ptr == first.get() ? 4096u : 8192u);
  }
  // Two live storages never share an address.
  EXPECT_NE(NeuronCachingAllocator::findPhysicalExtent(first.get()).address,
            NeuronCachingAllocator::findPhysicalExtent(second.get()).address);
}

TEST_F(AddressWordsTest, PhysicalExtentRefusesAnUnknownPointer) {
  EXPECT_THROW(NeuronCachingAllocator::findPhysicalExtent(reinterpret_cast<void*>(0xDEADBEEF)),
               std::exception);
}

TEST_F(AddressWordsTest, FillAddressWordsWritesLowThenHighAndRefusesOutsideTheStorage) {
  std::vector<uint32_t> words(8, 7);
  std::vector<at::neuron::AddressWordEntry> entries{
      {at::neuron::TensorDataRef{}, 2, 0x10, 0x20}, {at::neuron::TensorDataRef{}, 5, 0, 0x100}};
  at::neuron::FillAddressWords(words, entries, {{0x12'3456'7000ULL, 0x40}, {0x8000ULL, 0x100}});
  EXPECT_EQ(words, (std::vector<uint32_t>{7, 7, 0x3456'7010u, 0x12u, 7, 0x8000u, 0u, 7}));

  // A view past its storage, or words past the buffer, are refused.
  std::vector<at::neuron::AddressWordEntry> past{{at::neuron::TensorDataRef{}, 0, 0x30, 0x20}};
  EXPECT_THROW(at::neuron::FillAddressWords(words, past, {{0x1000, 0x40}}), std::exception);
  std::vector<at::neuron::AddressWordEntry> outside{{at::neuron::TensorDataRef{}, 7, 0, 4}};
  EXPECT_THROW(at::neuron::FillAddressWords(words, outside, {{0x1000, 0x40}}), std::exception);
}

TEST_F(AddressWordsTest, WordsAreFilledOnTheWorkerForTensorsAllocatedAfterQueuedWork) {
  // Nothing is synchronized between the allocations and the submission: the stream binds each
  // allocation when its worker reaches it, and the write's addresses resolve only then.
  auto words_dst = allocate(6 * sizeof(uint32_t));
  auto input = allocate(4096);
  auto output = allocate(2048);
  std::vector<uint32_t> words{1, 2, 0, 0, 0, 0};
  std::vector<at::neuron::AddressWordEntry> entries{
      {at::neuron::TensorDataRef{nullptr, input.get()}, 2, 0x40, 0x100},
      {at::neuron::TensorDataRef{nullptr, output.get()}, 4, 0, 2048}};
  auto kernel = std::make_unique<at::neuron::AddressWordsWriteKernelExecution>(
      "luminal::address_words", words, at::neuron::TensorDataRef{nullptr, words_dst.get()}, 0,
      std::move(entries), 0);
  at::neuron::SubmitOperationContext(at::neuron::getCurrentNeuronStream(0),
                                     std::make_unique<at::neuron::OperationContext>(
                                         std::move(kernel)));
  syncDevice();

  ASSERT_EQ(writes_.size(), 1u);
  EXPECT_EQ(writes_[0].tensor, NeuronCachingAllocator::findTensor(words_dst.get()));
  ASSERT_EQ(writes_[0].bytes.size(), 6 * sizeof(uint32_t));
  std::vector<uint32_t> written(6);
  std::memcpy(written.data(), writes_[0].bytes.data(), writes_[0].bytes.size());
  uint64_t in = ExpectedAddress(input.get(), 0x40), out = ExpectedAddress(output.get(), 0);
  EXPECT_EQ(written, (std::vector<uint32_t>{1, 2, static_cast<uint32_t>(in),
                                            static_cast<uint32_t>(in >> 32),
                                            static_cast<uint32_t>(out),
                                            static_cast<uint32_t>(out >> 32)}));
}

// ---------------------------------------------------------------------------
// Zero fills queued on the stream (zero_fill_neuron_bytes): device-to-device copies from a
// zeroed device block, byte-exact for any length.
// ---------------------------------------------------------------------------

// A fake device memory: every segment NRT allocates is host bytes at its VA, a slice is its
// segment's bytes at an offset, and nrta_tensor_write / nrta_tensor_copy move bytes in it.
class ZeroFillTest : public NeuronCachingAllocatorTest {
 protected:
  void SetUp() override {
    NeuronCachingAllocatorTest::SetUp();
    auto* nrt = MockNRT::GetInstance();
    EXPECT_CALL(*nrt, nrt_tensor_allocate(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke(
            [this](nrt_tensor_placement_t, int32_t, size_t size, const char*, nrt_tensor_t** out) {
              std::lock_guard<std::mutex> lock(mutex_);
              uintptr_t va = next_addr_;
              next_addr_ += std::max(size, static_cast<size_t>(0x1000));
              memory_[va] = std::vector<uint8_t>(size, 0xAB);  // never zero by itself
              *out = reinterpret_cast<nrt_tensor_t*>(va);
              return NRT_SUCCESS;
            }));
    EXPECT_CALL(*nrt, nrt_tensor_allocate_slice(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([this](const nrt_tensor_t* base, size_t offset, size_t,
                                      const char*, nrt_tensor_t** out) {
          // The worker's binding of a fresh allocation, made slow on purpose.
          std::this_thread::sleep_for(std::chrono::milliseconds(bind_delay_ms_.load()));
          *out = reinterpret_cast<nrt_tensor_t*>(reinterpret_cast<uintptr_t>(base) + offset);
          return NRT_SUCCESS;
        }));
    EXPECT_CALL(*nrt, nrta_tensor_write(_, _, _, _, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([this](nrt_tensor_t* tensor, const void* buf, uint64_t offset,
                                      uint64_t size, int, int, NRT_STATUS* ret, nrta_seq_t* seq) {
          std::lock_guard<std::mutex> lock(mutex_);
          std::memcpy(At(tensor, offset, size), buf, size);
          ++writes_;
          *ret = NRT_SUCCESS;
          *seq = 1;
          return NRT_SUCCESS;
        }));
    EXPECT_CALL(*nrt, nrta_tensor_copy(_, _, _, _, _, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([this](nrt_tensor_t* src, uint64_t src_offset, nrt_tensor_t* dst,
                                      uint64_t dst_offset, uint64_t size, int, int,
                                      NRT_STATUS* ret, nrta_seq_t* seq) {
          std::lock_guard<std::mutex> lock(mutex_);
          std::memmove(At(dst, dst_offset, size), At(src, src_offset, size), size);
          copies_.push_back(size);
          *ret = NRT_SUCCESS;
          *seq = 1;
          return NRT_SUCCESS;
        }));
  }

  // The fake bytes behind `tensor` + `offset`, checked to lie inside one segment.
  uint8_t* At(const nrt_tensor_t* tensor, uint64_t offset, uint64_t size) {
    uintptr_t va = reinterpret_cast<uintptr_t>(tensor) + offset;
    auto it = memory_.upper_bound(va);
    EXPECT_NE(it, memory_.begin());
    --it;
    EXPECT_LE(va + size, it->first + it->second.size()) << "access outside its segment";
    return it->second.data() + (va - it->first);
  }

  std::vector<uint8_t> Read(void* ptr, size_t size) {
    nrt_tensor_t* handle = NeuronCachingAllocator::findTensor(ptr);
    std::lock_guard<std::mutex> lock(mutex_);
    uint8_t* bytes = At(handle, 0, size);
    return std::vector<uint8_t>(bytes, bytes + size);
  }

  std::mutex mutex_;
  std::map<uintptr_t, std::vector<uint8_t>> memory_;
  std::atomic<int> bind_delay_ms_{0};
  int writes_ = 0;
  std::vector<uint64_t> copies_;
};

TEST_F(ZeroFillTest, ZeroesExactlyTheRequestedBytesOfAnyLength) {
  // More than two zero blocks (8 MiB) and not a multiple of 4 bytes, at an odd offset; queued
  // right after the allocation, with nothing synchronized.
  constexpr size_t kStorage = 24u << 20, kOffset = 5, kBytes = (17u << 20) + 3;
  auto storage = allocate(kStorage);
  torch_neuronx::utils::zero_fill_neuron_bytes(storage.get(), kOffset, kBytes, 0);
  syncDevice();
  auto bytes = Read(storage.get(), kStorage);
  for (size_t i = 0; i < kStorage; ++i) {
    const bool filled = i >= kOffset && i < kOffset + kBytes;
    ASSERT_EQ(bytes[i], filled ? 0 : 0xAB) << "byte " << i;
  }
  // One zero block, zeroed once by a write; then one copy per block-sized chunk.
  EXPECT_EQ(writes_, 1);
  EXPECT_EQ(copies_, (std::vector<uint64_t>{8u << 20, 8u << 20, (1u << 20) + 3}));

  // Later fills on the stream reuse the block; a zero-length fill queues nothing. (The block is
  // the process's, so this is one test: a second fixture would not hold its fake bytes.)
  copies_.clear();
  auto first = allocate(4096), second = allocate(4099);
  torch_neuronx::utils::zero_fill_neuron_bytes(first.get(), 0, 4096, 0);
  torch_neuronx::utils::zero_fill_neuron_bytes(second.get(), 0, 4099, 0);
  torch_neuronx::utils::zero_fill_neuron_bytes(second.get(), 0, 0, 0);
  syncDevice();
  EXPECT_EQ(Read(first.get(), 4096), std::vector<uint8_t>(4096, 0));
  EXPECT_EQ(Read(second.get(), 4099), std::vector<uint8_t>(4099, 0));
  EXPECT_EQ(writes_, 1);
  EXPECT_EQ(copies_, (std::vector<uint64_t>{4096, 4099}));

  // A fresh storage zero-filled and then handed on by its NRT tensor at once (transport's slab:
  // torch.zeros, then the handle): the fill is only queued, the binding is still on the worker,
  // and the host reads the handle after the binding, not before.
  bind_delay_ms_ = 200;
  auto fresh = allocate(3u << 20);
  torch_neuronx::utils::zero_fill_neuron_bytes(fresh.get(), 0, (3u << 20) - 1, 0);
  EXPECT_EQ(NeuronCachingAllocator::findTensor(fresh.get()), nullptr);  // not bound yet
  nrt_tensor_t* handle = NeuronCachingAllocator::findBoundTensor(fresh.get());
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle, NeuronCachingAllocator::findTensor(fresh.get()));
  bind_delay_ms_ = 0;
  syncDevice();
  auto fresh_bytes = Read(fresh.get(), 3u << 20);
  EXPECT_EQ(std::count(fresh_bytes.begin(), fresh_bytes.end(), 0), (3 << 20) - 1);
  EXPECT_EQ(fresh_bytes.back(), 0xAB);

  // The same ordering, many times, at varied binding delays and sizes (new segments and cached
  // blocks): allocation, zero fill, then the handle at once is always the bound storage's.
  std::vector<c10::DataPtr> live;
  for (int i = 0; i < 300; ++i) {
    bind_delay_ms_ = i % 7 == 0 ? 3 : 0;
    const size_t size = (i % 5 == 0) ? (size_t(1) << 20) + 4096 * (i % 13) : 512 + 61 * i;
    auto storage = allocate(size);
    torch_neuronx::utils::zero_fill_neuron_bytes(storage.get(), 0, size, 0);
    nrt_tensor_t* bound = NeuronCachingAllocator::findBoundTensor(storage.get());
    ASSERT_NE(bound, nullptr) << "iteration " << i;
    ASSERT_EQ(bound, NeuronCachingAllocator::findTensor(storage.get())) << "iteration " << i;
    if (i % 3 == 0) live.push_back(std::move(storage));  // the rest free and are reused
  }
  bind_delay_ms_ = 0;
  syncDevice();
}

// An allocation followed at once by a need for its NRT tensor (no fill, nothing synchronized),
// repeatedly, with the worker's binding slowed at random: the handle is always the bound one.
TEST_F(ZeroFillTest, AFreshAllocationsHandleIsReadAfterItsBinding) {
  std::vector<c10::DataPtr> live;
  for (int i = 0; i < 300; ++i) {
    bind_delay_ms_ = i % 4 == 0 ? 2 : 0;
    auto storage = allocate(256 + 97 * i);
    nrt_tensor_t* bound = NeuronCachingAllocator::findBoundTensor(storage.get());
    ASSERT_NE(bound, nullptr) << "iteration " << i;
    ASSERT_EQ(bound, NeuronCachingAllocator::findTensor(storage.get())) << "iteration " << i;
    if (i % 2 == 0) live.push_back(std::move(storage));
  }
  // Memory statistics and snapshots wait for every allocation already returned: right after
  // allocations (one a new segment), waitForPendingBindings leaves each bound.
  for (int i = 0; i < 100; ++i) {
    bind_delay_ms_ = i % 5 == 0 ? 2 : 0;
    auto a = allocate(300 + 7 * i);
    auto b = allocate(i % 10 == 0 ? (size_t(2) << 20) + 4096 * i : 1000 + i);
    NeuronCachingAllocator::waitForPendingBindings(i % 2 == 0 ? 0 : -1);
    ASSERT_NE(NeuronCachingAllocator::findTensor(a.get()), nullptr) << "iteration " << i;
    ASSERT_NE(NeuronCachingAllocator::findTensor(b.get()), nullptr) << "iteration " << i;
    if (i % 3 == 0) {
      live.push_back(std::move(a));
      live.push_back(std::move(b));
    }
  }
  bind_delay_ms_ = 0;
}
