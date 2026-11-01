import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'tools'))
from cue_reliability import differences as cue_differences
from verify_voice_design import differences as audition_differences, words


class WordEvaluationTests(unittest.TestCase):
    def compare(self, script, expected, heard):
        for compare in [cue_differences, audition_differences]:
            yield compare(words(expected), heard, True, line=script)

    def test_uncued_extra_he_is_not_a_reaction(self):
        for changes in self.compare('Please wait here.', 'Please wait here.', 'He, please wait here.'):
            self.assertTrue(changes)

    def test_required_he_and_laughter_words_are_not_deleted(self):
        for text in ['He waits here.', 'She laughs at the joke.', 'Laughter fills the room.']:
            for changes in self.compare('[laughs] ' + text, text, text):
                self.assertEqual(changes, [])

    def test_extra_laughter_is_allowed_only_with_a_reaction_cue(self):
        for changes in self.compare('[laughs] Please wait here.', 'Please wait here.', 'Ha ha! Please wait here.'):
            self.assertEqual(changes, [])
        for changes in self.compare('Please wait here.', 'Please wait here.', 'Ha ha! Please wait here.'):
            self.assertTrue(changes)

    def test_missing_words_repetition_and_negation_still_fail(self):
        for heard in ['No, he did not wait.', 'No, no, he did wait.', 'No, no, he did not wait again.']:
            for changes in self.compare('[laughs] No, no, he did not wait.', 'No, no, he did not wait.', heard):
                self.assertTrue(changes)

    def test_matching_annotation_does_not_hide_required_gasp(self):
        for changes in self.compare('[gasps] Please say gasp.', 'Please say gasp.', '*gasp* Please say.'):
            self.assertTrue(changes)
        for changes in self.compare('[gasps] Please say gasp.', 'Please say gasp.', '(gasps) Please say gasp.'):
            self.assertEqual(changes, [])

    def test_wrong_words_and_unrequested_annotations_still_fail(self):
        for script, heard in [('[gasps] The garden gate is closed.', '(gasps) The golden gate is closed.'),
                              ('[laughs] The garden gate is closed.', '*gasp* The garden gate is closed.')]:
            for changes in self.compare(script, 'The garden gate is closed.', heard):
                self.assertTrue(changes)


if __name__ == '__main__':
    unittest.main()
