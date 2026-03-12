# Contributing to Somnivex

Thanks for wanting to help. This is a solo side project so any contribution is genuinely appreciated.

---

## Getting started

```bash
git clone https://github.com/kosmickroma/somnivex
cd somnivex
pip install jax[cuda] flax pygame numpy
python main.py
```

Hit `S` to cycle regimes, `U`/`D` to rate, `Q` to quit.

---

## Ways to contribute

- **Add palettes** — just 4 RGB tuples in `nca/params.py`, no other files needed
- **Add GS regimes** — (f, k) pairs in `gs/engine.py`, reference: http://www.mrob.com/pub/comp/xmorphia
- **Bug reports** — open an issue with your OS, GPU, and full terminal output
- **Testing** — run it on your setup and report what happens, even "works fine" is useful
- **Bigger stuff** — check the open issues, anything labeled `help wanted` is fair game

---

## Submitting a PR

1. Fork the repo
2. Make your changes on a new branch
3. Test it runs — `python main.py` should launch without errors
4. Open a pull request with a short description of what you changed and why

No formal review process, no style guide. Just make sure it runs and describe what you did.

---

## Questions

Open an issue or email kosmickroma@gmail.com
