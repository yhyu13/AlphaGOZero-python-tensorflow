Simple Python Elo rating
########################

Very simple Python implementation of the Elo rating system.

It contains two functions: ``expected(A, B)``, to calculate the
expected score of ``A`` in a match against ``B``; and ``elo(...)`` to
calculate the new score for each player.


Usage example
=============

(`Example from Wikipedia <http://en.wikipedia.org/wiki/Elo_rating_system>`_)

In a five-round tournament, player A, with a rating of ``1613``, plays
against opponents with the following ratings: ``1609``, ``1477``,
``1388``, ``1586``, ``1720``.

The expected score is therefore:

.. code-block:: python

    from elo import expected

    exp  = expected(1613, 1609)
    exp += expected(1613, 1477)
    exp += expected(1613, 1388)
    exp += expected(1613, 1586)
    exp += expected(1613, 1720)

    # exp == 2.867

A lost match #1, draw match #2, wins #3 and #4 and loses #5.
Therefore the player's actual score is ``(0 + 0.5 + 1 + 1 + 0) = 2.5``.

We can now use this to calculate the new Elo rating for A:

.. code-block:: python

    from elo import elo

    elo(1613, 2.867, 2.5, k=32)  # 1601
