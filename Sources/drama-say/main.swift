import DramaMCP
import Foundation

exit(await SayCommand.main(Array(CommandLine.arguments.dropFirst()), SayEnvironment(api: HTTPDramaAPI(), play: { try await Speakers.play($0) })))
