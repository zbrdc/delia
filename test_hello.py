import subprocess
import unittest
class TestHelloWorld(unittest.TestCase):
    def test_output(self):
        result = subprocess.run(['./hello_world.py'], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), 'Hello World')
        self.assertEqual(result.returncode, 0)
if __name__ == '__main__':
    unittest.main()