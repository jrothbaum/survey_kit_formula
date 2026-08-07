from polars_formula.parser.args import Arg, split_args


def test_simple_positional():
    assert split_args("x") == [Arg(None, "x")]


def test_positional_and_keyword():
    assert split_args("x, degree=2") == [Arg(None, "x"), Arg("degree", "2")]


def test_nested_call_in_arg_not_split():
    assert split_args("y, contr.treatment(base=2)") == [
        Arg(None, "y"),
        Arg(None, "contr.treatment(base=2)"),
    ]


def test_string_literal_with_comma_not_split():
    assert split_args("x, 'a, b'") == [Arg(None, "x"), Arg(None, "'a, b'")]


def test_double_equals_is_not_keyword():
    assert split_args("x == 1") == [Arg(None, "x == 1")]


def test_whitespace_stripped():
    assert split_args(" x , degree = 2 ") == [Arg(None, "x"), Arg("degree", "2")]


def test_empty_args():
    assert split_args("") == []
