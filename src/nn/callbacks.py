from catalyst.dl.callbacks import MeterMetricsCallback as CatalystMeterMetricsCallback,\
                                AUCCallback as CatalystAUCCallback,\
                                PrecisionRecallF1ScoreCallback as CatalystPrecisionRecallF1ScoreCallback,\
                                ConfusionMatrixCallback

class MeterMetricsCallback(CatalystMeterMetricsCallback):

    def on_batch_end(self, state):
        logits = state.output[self.output_key].detach().float()
        targets = state.input[self.input_key].detach().float()
        probabilities = self.activation_fn(logits)

        for i in range(self.num_classes):
            target = (targets == i).float()
            self.meters[i].add(probabilities[:, i], target)

class AUCCallback(MeterMetricsCallback, CatalystAUCCallback):
    pass

class PrecisionRecallF1ScoreCallback(MeterMetricsCallback, CatalystPrecisionRecallF1ScoreCallback):
    pass
