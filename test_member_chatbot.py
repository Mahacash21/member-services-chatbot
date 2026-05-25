from member_chatbot import validate_question, check_rate_limit
from member_chatbot import get_cache_key, cache_response
from member_chatbot import get_cached_response

# ── Test validate_question ─────────────────────────────
def test_empty_question():
    valid, msg = validate_question("")
    assert valid == False
    assert "empty" in msg.lower()
    print("✅ test_empty_question passed")

def test_valid_question():
    valid, msg = validate_question("What is my deductible?")
    assert valid == True
    assert msg is None
    print("✅ test_valid_question passed")

def test_too_long_question():
    valid, msg = validate_question("x" * 501)
    assert valid == False
    assert "long" in msg.lower()
    print("✅ test_too_long_question passed")

def test_phi_detection():
    valid, msg = validate_question("My SSN is 123-45-6789")
    assert valid == False
    print("✅ test_phi_detection passed")

def test_short_question():
    valid, msg = validate_question("Hi")
    assert valid == False
    print("✅ test_short_question passed")

# ── Test rate limiting ─────────────────────────────────
def test_rate_limit_allows_normal_use():
    allowed = check_rate_limit("test_user_1", max_requests=5)
    assert allowed == True
    print("✅ test_rate_limit_allows passed")

def test_rate_limit_blocks_excessive_use():
    for _ in range(10):
        check_rate_limit("test_user_2", max_requests=5)
    blocked = check_rate_limit("test_user_2", max_requests=5)
    assert blocked == False
    print("✅ test_rate_limit_blocks passed")

# ── Test caching ───────────────────────────────────────
def test_cache_stores_and_retrieves():
    cache_response("What is my deductible?", "Your deductible is $1500")
    cached = get_cached_response("What is my deductible?")
    assert cached == "Your deductible is $1500"
    print("✅ test_cache_stores_and_retrieves passed")

def test_cache_key_case_insensitive():
    key1 = get_cache_key("What is my deductible?")
    key2 = get_cache_key("WHAT IS MY DEDUCTIBLE?")
    assert key1 == key2
    print("✅ test_cache_key_case_insensitive passed")

# ── Run all tests ──────────────────────────────────────
if __name__ == "__main__":
    print("Running tests...\n")
    test_empty_question()
    test_valid_question()
    test_too_long_question()
    test_phi_detection()
    test_short_question()
    test_rate_limit_allows_normal_use()
    test_rate_limit_blocks_excessive_use()
    test_cache_stores_and_retrieves()
    test_cache_key_case_insensitive()
    print("\n✅ All tests passed!")