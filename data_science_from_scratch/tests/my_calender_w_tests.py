import unittest
import requests # with mocking the entire lib, you don't even need to actually import it
from requests.exceptions import Timeout, ConnectTimeout
from unittest.mock import patch, Mock

# Mock requests to control its behavior
requests = Mock()

def get_holidays():
    r = requests.get('http://localhost/api/holidays')
    if r.status_code == 200:
        return r.json()
    return None


class TestCalendar(unittest.TestCase):

    def test_get_holidays_timeout_with_side_effect(self):
        requests.get.side_effect = Timeout
        with self.assertRaises(Timeout):
            get_holidays()

    def log_request(self, url):
        # Log a fake request for test output purposes
        print(f'Making a request to {url}.')
        print('Request received!')

        # Create a new Mock to imitate a Response
        response_mock = Mock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            '12/25': 'Christmas',
            '7/4': 'Independence Day',
        }
        return response_mock

    def test_get_holidays_logging(self):
        # Test a successful, logged request
        requests.get.side_effect = self.log_request
        assert get_holidays()['12/25'] == 'Christmas'



if __name__ == '__main__':
    unittest.main()
