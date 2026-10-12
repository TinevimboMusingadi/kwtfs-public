from kwtfs import checks


def test_finds_false_arithmetic_from_earlier_run():
  bad = checks.false_claims("<mode:cot>\n(7212*7213)+59=79981\n<mode:jev>\nB")
  assert len(bad) == 1 and bad[0].actual == 7212 * 7213 + 59


def test_true_lines_pass():
  trace = "4 x 2 = 8 bushels\n3 x 2 = 6\n8 + 6 + 21 = 35\n$200 - $198 = $2\n(4682*2)+78=9442"
  cs = checks.claims(trace)
  assert len(cs) >= 4 and checks.faithful(trace)


def test_formats_and_edge_cases():
  assert checks.faithful("1,200 / 4 = 300")
  assert not checks.faithful("12 × 12 = 140")
  assert checks.faithful("10 / 3 = 3.33")  # rounding to 2 dp is fine
  assert checks.faithful("no arithmetic here, just words = 5")
  assert checks.faithful("x = 5")  # a definition, not a checkable claim
  assert checks.faithful("2**999999 = 1")  # refuses huge powers instead of hanging
  assert checks.evaluate("__import__('os')") is None
