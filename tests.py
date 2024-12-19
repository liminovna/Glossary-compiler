from classes import GlossaryCompiler
import unittest


class TestGlossaryCompilerConstructorExt(unittest.TestCase):
    def test_gcomp_extension_txt(self):
        self.assertEqual(GlossaryCompiler(filename='Alice_in_Wonderland.txt', lang='en').extension, 'txt')

    def test_gcomp_extension_pdf(self):
        self.assertEqual(GlossaryCompiler(filename='Alice_in_Wonderland.pdf', lang='en').extension, 'pdf')

class TestGlossaryCompilerConstructorLang(unittest.TestCase):
    def test_gcomp_extension_txt(self):
        self.assertEqual(GlossaryCompiler(filename='Alice_in_Wonderland.txt', lang='en').lang, 'en')

    def test_gcomp_extension_pdf(self):
        self.assertEqual(GlossaryCompiler(filename='Alice_in_Wonderland.pdf', lang='ru').lang, 'ru')

if __name__ == '__main__':
    unittest.main()