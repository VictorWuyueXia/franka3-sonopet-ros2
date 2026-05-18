from fr3_sonopet_supervisor.operator_gates import is_execute_token_valid


def test_execute_token_requires_exact_command_after_trim():
    assert is_execute_token_valid(" EXECUTE ")
    assert not is_execute_token_valid("execute")

