class Analyzer(object):
    def __init__(self, database, evaluator):
        self.database = database
        self.evaluator = evaluator

    def test(self, limit = None):
        entries = self.database.to_list()
        if limit:
            entries.sort(key=lambda e: e.score, reverse=True)
            entries = entries[:limit]
        for entry in entries:
            entry.score = self.evaluator.eval(entry.model_state)
        return entries