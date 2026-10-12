from kwtfs import format as fmt


def test_parse_modes():
  p = fmt.parse("<mode:jev>\nB", 4)
  assert p.valid and p.via_jev and p.answer == "B" and not p.switched
  p = fmt.parse("<mode:cot>\n3+4=7\n<mode:jev>\nC", 4)
  assert p.valid and p.switched and p.modes == ("cot", "jev")
  p = fmt.parse("<mode:direct>\n<answer>7</answer>", 0)
  assert p.valid and p.answer == "7" and not p.via_jev


def test_parse_rejects_off_format():
  assert not fmt.parse("<mode:jev>\nB", 0).valid  # no options shown
  assert not fmt.parse("<mode:jev>\nE", 4).valid  # letter out of range
  assert not fmt.parse("<mode:direct>\n<mode:jev>\nA", 4).valid  # only cot may hand off
  assert not fmt.parse("Sure! <mode:direct>\n<answer>7</answer>", 0).valid
  assert not fmt.parse("<mode:cot>\n<answer>1</answer><answer>2</answer>", 0).valid
  assert not fmt.parse("<mode:direct>\n" + "x" * 200 + "<answer>7</answer>", 0).valid


def test_correctness_is_exact():
  opts = ["10", "100", "1000"]
  assert fmt.is_correct_exact(fmt.parse("<mode:jev>\nB", 3), "100", opts)
  assert not fmt.is_correct_exact(fmt.parse("<mode:jev>\nC", 3), "100", opts)
  # no substring credit
  assert not fmt.is_correct_exact(fmt.parse("<mode:direct>\n<answer>1000</answer>", 0), "100", [])
  assert fmt.is_correct_exact(fmt.parse("<mode:direct>\n<answer>1,000</answer>", 0), "1000", [])
  assert fmt.is_correct(fmt.parse("<mode:direct>\n<answer>1,000</answer>", 0), "1000", [])


def test_genre_and_control():
  p = fmt.parse_tagged("<genre:arith><mode:cot>\n2+2=4\n<answer>4</answer>", 0, "cot")
  assert p.valid and p.genre == "arith" and p.trace.answer == "4"
  assert not fmt.parse_tagged("<genre:arith><mode:direct>\n<answer>4</answer>", 0, "cot").valid
  p = fmt.parse_tagged("<mode:jev>\nB", 3, "auto")  # genre tag optional for validity
  assert p.valid and p.genre is None
  assert fmt.parse_tagged("<genre:bogus><mode:jev>\nB", 3).genre is None
  # forced CoT with no options must end in <answer>, never a JEV hand-off
  assert not fmt.parse_tagged("<genre:arith><mode:cot>\n2+2=4\n<mode:jev>\nA", 0, "cot").valid
  assert "Mode: cot." in fmt.chat_prompt("q", "cot")
  assert fmt.chat_prompt("q").endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_letter_answers_count_with_options():
  opts = ["low", "high"]
  p = fmt.parse_tagged("<genre:decision><mode:direct>\n<answer>B</answer>", 2, "direct").trace
  assert fmt.is_correct(p, "high", opts) and not fmt.is_correct(p, "low", opts)
  assert not fmt.is_correct_exact(p, "high", opts)  # the exact rule alone gives no letter credit
  p = fmt.parse_tagged("<genre:arith><mode:direct>\n<answer>B</answer>", 0).trace
  assert not fmt.is_correct(p, "high", [])  # no options: a letter is just text
