import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def mock_env():
    with patch.dict(os.environ, {
        "MONGODB_URI": "mongodb://localhost:27017/",
        "HF_TOKEN": "test_hf_token",
        "GEMINI_API_KEY": "test_gemini_api_key",
        "PORT": "8080",
        "API_TOKEN": "test_api_token",
        "VALID_CREDENTIALS": '{"test_user": "test_password"}'
    }):
        yield

# Mock streamlit before importing the services
sys.modules['streamlit'] = MagicMock()

@pytest.fixture
def mock_streamlit():
    with patch('streamlit.session_state', new_callable=MagicMock) as mock_session_state:
        with patch('streamlit.rerun') as mock_rerun:
            yield mock_session_state, mock_rerun

def test_check_authentication_not_authenticated(mock_streamlit, mock_env):
    from src.web_app.services.auth_service import check_authentication
    mock_session_state, _ = mock_streamlit
    # Simulate that 'authenticated' is not in session_state
    mock_session_state.__contains__.return_value = False
    assert not check_authentication()

def test_check_authentication_authenticated(mock_streamlit, mock_env):
    from src.web_app.services.auth_service import check_authentication
    mock_session_state, _ = mock_streamlit
    # Simulate that 'authenticated' is in session_state and is True
    mock_session_state.__contains__.return_value = True
    mock_session_state.authenticated = True
    assert check_authentication()

def test_logout(mock_streamlit, mock_env):
    from src.web_app.services.auth_service import logout
    mock_session_state, mock_rerun = mock_streamlit

    # Set attributes that should be deleted by logout
    mock_session_state.authenticated = True
    mock_session_state.username = "test"
    mock_session_state.chats = {"chat_1": []}
    mock_session_state.current_chat_id = "chat_1"
    mock_session_state.chat_counter = 1

    logout()

    # Assert that authenticated is set to False
    assert not mock_session_state.authenticated

    # Assert that the correct keys were popped from the session state
    mock_session_state.pop.assert_any_call('username', None)
    mock_session_state.pop.assert_any_call('chats', None)
    mock_session_state.pop.assert_any_call('current_chat_id', None)
    mock_session_state.pop.assert_any_call('chat_counter', None)

    mock_rerun.assert_called_once()
