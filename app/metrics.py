class TopicDiversity():
    def __init__(self, topk=10):
        """
        Initialize metric

        Parameters
        ----------
        topk: top k words on which the topic diversity will be computed
        """
        self.topk = topk

    def info(self):
        return {
            "name": "Topic diversity"
        }

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        td : score
        """
        topics = model_output["topics"]
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than ' + str(self.topk))
        else:
            unique_words = set()
            for topic in topics:
                unique_words = unique_words.union(set(topic[:self.topk]))
            td = len(unique_words) / (self.topk * len(topics))
            return td
