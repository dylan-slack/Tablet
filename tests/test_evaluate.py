#  MIT License
#
#  Copyright (c) 2023.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#

from unittest import TestCase

from Tablet import evaluate


class TestEvaluator(TestCase):
    def test_get_test_hf_datasets(self):
        benchmark_path = "./data/benchmark/performance/"
        tasks = ['A37/prototypes-synthetic-performance-3',
                 'Adult/prototypes-synthetic-performance-3']
        evaluator = evaluate.Evaluator(benchmark_path=benchmark_path,
                                       tasks_to_run=tasks,
                                       encoding_format="flan",
                                       k_shot=0)
        whooping_cough, adult = evaluator.get_test_hf_datasets()
        assert len(whooping_cough["text"]) == len(whooping_cough["label"])

    def test_eval(self):
        benchmark_path = "./data/benchmark/performance/"
        tasks = ['A37/prototypes-synthetic-performance-3']
        evaluator = evaluate.Evaluator(benchmark_path=benchmark_path,
                                       tasks_to_run=tasks,
                                       encoding_format="flan",
                                       model="google/flan-t5-small",
                                       k_shot=0)
        evaluator.run_eval(how_many=1)
