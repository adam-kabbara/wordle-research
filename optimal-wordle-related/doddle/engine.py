from __future__ import annotations

from dataclasses import dataclass

from .exceptions import FailedToFindASolutionError
from .game import Game, SimultaneousGame
from .guess import Guess
from .histogram import HistogramBuilder
from .scoring import Scorer
from .simul_solver import SimulSolver
from .solver import Solver
from .views import RunReporter
from .words import Dictionary, Word


@dataclass
class Engine:
    """Primary class for running a Doddle game."""

    dictionary: Dictionary
    scorer: Scorer
    histogram_builder: HistogramBuilder
    solver: Solver[Guess]
    reporter: RunReporter

    def run(self, solution: Word, user_guesses: list[Word]) -> list:
        """Runs a Doddle game.

        Args:
            solution (Word): The solution
            user_guesses (list[Word]): A list of user-supplied, opening guesses

        Raises:
            FailedToFindASolutionError: If no solution is found

        Returns:
            Game: A Game object summarising the simulation.
        """

        all_words, available_answers = self.dictionary.words
        game = Game(available_answers, solution, user_guesses)
        guess = game.user_guess(0) or self.solver.seed(all_words.word_length)
        optimal_solution = [guess]
        MAX_ITERS = 20
        for i in range(1, MAX_ITERS + 1):
            histogram = self.histogram_builder.get_solns_by_score(available_answers, guess)
            score = self.scorer.score_word(solution, guess)
            available_answers = histogram[score]

            # update(n: int, the round, guess: Word, score: int, potential_solns: WordSeries)
            game.update(i, guess, score, available_answers)
            # self.reporter.display(game)

            if game.is_solved:
                # print(optimal_solution)
                return optimal_solution[:]

            guess = game.user_guess(i) or self.solver.get_best_guess(all_words, available_answers).word
            optimal_solution.append(guess)

        raise FailedToFindASolutionError(f"Failed to converge after {MAX_ITERS} iterations.")
    
    def my_run(self, solution: Word, user_guesses: list[Word]) -> list:
        """Runs a Doddle game.

        Args:
            solution (Word): The solution
            user_guesses (list[Word]): A list of user-supplied, opening guesses

        Raises:
            FailedToFindASolutionError: If no solution is found

        Returns:
            Game: A Game object summarising the simulation.
        """

        all_words, available_answers = self.dictionary.words
        game = Game(available_answers, solution, user_guesses)
        guess = game.user_guess(0) or self.solver.seed(all_words.word_length)
        res = []
        terminate = False

        MAX_ITERS = 20
        for i in range(1, MAX_ITERS + 1):
            histogram = self.histogram_builder.get_solns_by_score(available_answers, guess)
            score = self.scorer.score_word(solution, guess)
            available_answers = histogram[score]

            # update(n: int, the round, guess: Word, score: int, potential_solns: WordSeries)
            # update(n: int, the round, guess: Word, score: int, potential_solns: WordSeries)
            game.update(i, guess, score, available_answers)
            # self.reporter.display(game)
            if terminate and game.is_solved:
                res.append((guess, 0))
                return res

            elif terminate:
                res.append((guess, len(available_answers)))
                return res

            if game.is_solved:
                return res
            
            prev_guess = guess
            guess = game.user_guess(i) or self.solver.get_best_guess(all_words, available_answers).word
            if not game.user_guess(i):
                res.append((prev_guess, len(available_answers)))
                terminate = True
        raise FailedToFindASolutionError(f"Failed to converge after {MAX_ITERS} iterations.")


@dataclass
class SimulEngine:
    """Primary class for running a simultaneous Doddle game."""

    dictionary: Dictionary
    scorer: Scorer
    histogram_builder: HistogramBuilder
    solver: SimulSolver[Guess, Guess]
    reporter: RunReporter

    def run(self, solns: list[Word], user_guesses: list[Word]) -> SimultaneousGame:
        """Runs a simultaneous Doddle game.

        Args:
            solns (list[Word]): The solutions to each game
            user_guesses (list[Word]): The list of user-supplied, opening guesses

        Raises:
            FailedToFindASolutionError: If no solution is found

        Returns:
            SimultaneousGame: A SimultaneousGame object summarising the simulation
        """
        all_words, common_words = self.dictionary.words
        simul_game = SimultaneousGame(common_words, solns, user_guesses)
        guess = simul_game.user_guess(0) or self.solver.seed(all_words.word_length)

        MAX_ITERS = 20 + len(solns)
        for i in range(1, MAX_ITERS + 1):
            for game in simul_game:
                if game.is_solved:
                    continue
                available_answers = game.potential_solns
                histogram = self.histogram_builder.get_solns_by_score(available_answers, guess)
                score = self.scorer.score_word(game.soln, guess)
                new_available_answers = histogram[score]
                simul_game.update(i, game, guess, score, new_available_answers)

            self.reporter.display(simul_game)

            if simul_game.is_solved:
                return simul_game

            guess = simul_game.user_guess(i) or self.solver.get_best_guess(all_words, simul_game).word

        raise FailedToFindASolutionError(f"Failed to converge after {MAX_ITERS} iterations.")
