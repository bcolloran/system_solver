# General guide for agents working on the param_solver crate

- Familiarize yourself with the high level design objectives in README.md
- When in a chat session, prefer to provide complete code snippets rather than diffs, unless specifically asked for a diff. Moreover, make sure the code snippets come in easily copy-pastable chunks -- complete files are ideal, but short of that, focus on providing complete functions or logical blocks of code. A small snippet within a large code block is hard to use, and they often lead to mistakes.
- As an LLM, you are prone to sycophantic behavior, and may try to please the user by agreeing with them even when they are wrong. Be vigilant against this tendency, and always prioritize correctness over pleasing the user. When in doubt, perform online research to verify facts and provide citations to help explain to the user why they may be mistaken.

# Tests
- When writing tests for physical dynamics, avoid hardcoding expected numerical values for arbitrary configurations. These are extremely hard for a human to understand and interpret, as they often require doing mental computations. Instead:
  - specify equilibrium or edge cases where expected values can be derived analytically for simple cases and special constants (0, 1, -1, pi, etc). A test like "at equilibrium, some quantity should be zero" is easy to understand and verify.
  - encode system invariants and properties that should hold true regardless of specific numbers with inequality checks or relative comparisons. For example, "after applying a force, the velocity should increase" is easy to understand and verify.
    - Inequalities relative to zero (sign checks) are particularly easy to understand.
    - Relative checks on monotone functions are pretty good.
    - Checks relative to special and edge case values (0, 1, -1, pi, etc) are also good.