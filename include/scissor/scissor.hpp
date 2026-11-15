#ifndef SCISSOR_HPP
#define SCISSOR_HPP

// ======================================================
// scissor/scissor.hpp
//
// Umbrella header: include this single file to get the
// complete scissor public API.
//
// Public API surface:
//   scissor::Segment, scissor::IReader            (io_reader.hpp)
//   scissor::TransferType, scissor::FileSrc,
//   scissor::Config, scissor::FileRequest,
//   scissor::LoadHandle                           (types.hpp)
//   scissor::common::Batch                        (batch.hpp)
//   scissor::AggregatorResult, scissor::Aggregator (aggregator.hpp)
//   scissor::RegressorConfig, scissor::Regressor  (regressor.hpp)
//   scissor::Loader                               (loader.hpp)
//   scissor::LiburingReader, scissor::PosixReader,
//   scissor::MmapReader                           (readers.hpp)
//   SCISSOR_* log macros                          (log.hpp)
//
// Note: batch.hpp is included via loader.hpp (Loader::next() returns
// common::Batch*).  Other src/common/ headers remain internal.
// ======================================================

#include <scissor/io_reader.hpp>
#include <scissor/types.hpp>
#include <scissor/aggregator.hpp>
#include <scissor/regressor.hpp>
#include <scissor/loader.hpp>
#include <scissor/readers.hpp>
#include <scissor/log.hpp>

#endif // SCISSOR_HPP
