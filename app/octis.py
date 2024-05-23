from abc import ABC, abstractmethod


class AbstractMetric(ABC):
    """
    Class structure of a generic metric implementation
    """

    def __init__(self):
        """
        init metric
        """
        pass

    @abstractmethod
    def score(self, model_output):
        """
        Retrieves the score of the metric

        :param model_output: output of a topic model in the form of a dictionary. See model for details on
        the model output
        :type model_output: dict
        """
        pass

    def get_params(self):
        return [att for att in dir(self) if not att.startswith("_") and att != 'info' and att != 'score' and
                att != 'get_params']


class TopicDiversity(AbstractMetric):
    def __init__(self, topk=10):
        """
        Initialize metric

        Parameters
        ----------
        topk: top k words on which the topic diversity will be computed
        """
        AbstractMetric.__init__(self)
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
