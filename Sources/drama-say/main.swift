// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaMCP
import Foundation

exit(await SayCommand.main(Array(CommandLine.arguments.dropFirst()), SayEnvironment(api: HTTPDramaAPI(), play: { try await Speakers.play($0) })))
