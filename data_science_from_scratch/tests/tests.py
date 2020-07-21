import unittest
from unittest.mock import patch, Mock, create_autospec
from requests.exceptions import Timeout, ConnectionError
import my_calendar
from my_calendar import requests, get_holidays


class TestCalendar(unittest.TestCase):

    # patch() replaces the real objects in your code with Mock instances
    @patch.object(requests, 'get', side_effect=Timeout)
    def test_get_holidays_timeout(self, mocked_get):
        with self.assertRaises(Timeout):
            get_holidays()

    @patch('my_calendar.requests')
    def test_get_holidays_timeout_mock_lib(self, mock_requests):
        mock_requests.get.side_effect = Timeout
        with self.assertRaises(Timeout):
            get_holidays()
            mock_requests.get.assert_called_once()

    def test_get_holidays_timeout_patch(self):
        with patch("my_calendar.requests") as mocked_requests:
            mocked_requests.get.side_effect = Timeout
            with self.assertRaises(Timeout):
                get_holidays()
                mocked_requests.get.assert_called_once()

    def test_non_able_to_patch_local_import(self):
        # from my_calendar import get_holidays binds the real function to the local scope. so, even though, patch()
        # with path to the function, it won't create a mock to replace, bc local referrence to the unmocked object already exists
        with patch("my_calendar.get_holidays"):
            with self.assertRaises(ConnectionError):
                get_holidays()

    def test_autospec_method_accessibility(self):
        # create_autospec implement automatic spefications, with autospec=True, close to set explicitly as below
        # calendar = Mock(spec=['is_weekday', 'get_holidays'])
        with patch('__main__.my_calendar', autospec=True) as calendar:
            calendar.is_weekday()
            with self.assertRaises(AttributeError):
                calendar.not_available_func()

if __name__ == '__main__':
    unittest.main()
