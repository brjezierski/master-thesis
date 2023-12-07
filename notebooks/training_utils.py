from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import custom_losses
import custom_evaluators
from sentence_transformers import SentenceTransformer, InputExample
import model_freeze as freeze
import prediction_helpers as ph
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import losses
from sentence_transformers import evaluation
import torch
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import spearmanr
from classification_training_utils import prepare_for_training as prepare_for_classification_training
import tensorflow as tf

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print(torch.cuda.device_count())
print("Device: ", device)


def create_weighted_avg(df, column_list, proportion_list, new_col_name):
    column_list = [f'{col}_similarity' for col in column_list]
    column_list.insert(0, 'similarity')

    # Proportion is a tuple of weights
    column_sum = 0.0
    for column, proportion in zip(column_list, proportion_list):
        column_sum += df[column] * proportion
    df[new_col_name] = column_sum / sum(proportion_list)
    # print(f'Proportions: {[p/sum(proportion_list)*SBERT_DIM for p in proportion_list]}')
    return df


def normalize_column(df, column_name):
    scaler = MinMaxScaler()
    # Reshape the data because the scaler function expects it
    df[column_name] = scaler.fit_transform(
        df[column_name].values.reshape(-1, 1))
    return df


def get_data(column_list, column_proportions, params, current_prefix='thesis/sbert-all/'):
    test_df = pd.read_csv(f'{current_prefix}test.tsv', sep='\t', index_col=0)
    val_df = pd.read_csv(f'{current_prefix}val.tsv', sep='\t', index_col=0)
    train_df = pd.read_csv(f'{current_prefix}train.tsv', sep='\t', index_col=0)

    if params["USE_WEIGHTED_SIMILARITY"]:
        for column in column_list:
            if '-inverse' in column:
                similarity_column = column.replace(
                    '-inverse', '') + '_similarity'
                train_df[similarity_column] = 1 - train_df[similarity_column]
                val_df[similarity_column] = 1 - val_df[similarity_column]
                test_df[similarity_column] = 1 - test_df[similarity_column]

        column_list = [column.replace('-inverse', '')
                       for column in column_list]
        test_df = create_weighted_avg(
            test_df, column_list, column_proportions, 'weighted_similarity')
        val_df = create_weighted_avg(
            val_df, column_list, column_proportions, 'similarity')
        train_df = create_weighted_avg(
            train_df, column_list, column_proportions, 'similarity')

    if params["USE_MANUAL_LABELS"]:
        test_df = normalize_column(test_df, 'label')
        val_df = normalize_column(val_df, 'label')
    else:
        test_df['label'] = test_df['similarity']
        val_df['label'] = val_df['similarity']

    return train_df, val_df, test_df


def callback(score, epoch, steps):
    print(f"Score at epoch {epoch}, step {steps}: {score}")


def convert_to_list(df, similarity_metrics, training_mode=True):
    df_json = pd.DataFrame()
    print(similarity_metrics)

    def get_embeddings(row, training_mode):
        embeddings = []
        for col in similarity_metrics:
            embeddings.append(row[col])
        if not training_mode:
            embeddings.append(row['label'])
        return embeddings

    df_json['data'] = df.apply(lambda row: [
                               row['snippet1'], row['snippet2'], get_embeddings(row, training_mode)], axis=1)
    out_list = df_json['data'].tolist()
    return out_list


def prepare_data(train_df, val_df, similarity_metrics):
    val_list = convert_to_list(val_df, similarity_metrics, training_mode=False)
    train_list = convert_to_list(train_df, similarity_metrics)

    train_examples = []
    for sample in train_list:
        ex = InputExample(texts=[sample[0], sample[1]], label=sample[2])
        train_examples.append(ex)

    dev_examples = []
    for sample in val_list:
        ex = InputExample(texts=[sample[0], sample[1]], label=sample[2])
        dev_examples.append(ex)

    return train_examples, dev_examples


def train(model, params, train_df, val_df):
    if "distill" in params["LOSSES"]:
        similarity_metrics = [
            f'{col}_similarity' for col in params["FEATURES"][2:]]
        train_examples_distil_loss, dev_examples_distil_loss = prepare_data(
            train_df, val_df, similarity_metrics)
    if "distill2" in params["LOSSES"]:
        similarity_metrics = [
            f'{col}_similarity' for col in params["FEATURES"][2:]] + ['similarity']
        train_examples_distil_loss, dev_examples_distil_loss = prepare_data(
            train_df, val_df, similarity_metrics)
    if "consistency" in params["LOSSES"]:
        similarity_metrics = [
            f'{col}_similarity' for col in params["FEATURES"][2:]]
        train_examples_consistency_loss, dev_examples_consistency_loss = prepare_data(
            train_df, val_df, similarity_metrics)
    train_examples_cos_sim_loss, dev_examples_cos_sim_loss = prepare_data(
        train_df, val_df, ['similarity'])

    train_objectives = []

    if "consistency" in params["LOSSES"]:
        # init model for consistency, parameters are frozen
        teacher = SentenceTransformer(params["SBERT_INIT"], device="cuda")
        freeze.freeze_all_layers(teacher)

        train_dataloader_consistency_loss = torch.utils.data.DataLoader(
            train_examples_consistency_loss, shuffle=True, batch_size=params["BATCH_SIZE"])
        consistency_loss = custom_losses.MultipleConsistencyLoss(
            model, teacher)
        train_objectives.append(
            (train_dataloader_consistency_loss, consistency_loss))

    if params["USE_DISTILL"]:
        train_dataloader_distil_loss = torch.utils.data.DataLoader(
            train_examples_distil_loss, shuffle=True, batch_size=params["BATCH_SIZE"])
        dev_dataloader_distil_loss = torch.utils.data.DataLoader(
            dev_examples_distil_loss, shuffle=False, batch_size=params["BATCH_SIZE"])

        loss_model = custom_losses.DistilLoss if "distill" in params[
            "LOSSES"] else custom_losses.DistilLoss2
        distill_loss_training = loss_model(model,
                                           sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                           num_labels=params["N"] + 1,
                                           feature_dim=params["FEATURE_DIM"],
                                           bias_inits=None)
        distill_loss_validation = loss_model(model,
                                             sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                             training_mode=False,
                                             # TODO not sure
                                             num_labels=params["N"] + 1,
                                             feature_dim=params["FEATURE_DIM"],
                                             bias_inits=None)
        train_objectives.append(
            (train_dataloader_distil_loss, distill_loss_training))
        evaluator = custom_evaluators.Distil2Evaluator(
            dev_dataloader_distil_loss, loss_model_distil=distill_loss_validation)

    if "cosine" in params["LOSSES"]:
        train_dataloader_cos_sim_loss = torch.utils.data.DataLoader(
            train_examples_cos_sim_loss, shuffle=True, batch_size=params["BATCH_SIZE"])
        dev_dataloader_cos_sim_loss = torch.utils.data.DataLoader(
            dev_examples_cos_sim_loss, shuffle=False, batch_size=params["BATCH_SIZE"])
        cos_sim_loss = losses.CosineSimilarityLoss(model=model)
        train_objectives.append((train_dataloader_cos_sim_loss, cos_sim_loss))
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            val_df['snippet1'].tolist(), val_df['snippet2'].tolist(), val_df['label'].tolist(), show_progress_bar=True)

    # train_loss = losses.ContrastiveLoss(model=model)
    # teacher_loss_training = custom_losses.MultipleConsistencyLoss(model, teacher)

    if "consistency" in params["LOSSES"] and params["USE_DISTILL"]:
        teacher_loss_validation = custom_losses.MultipleConsistencyLoss(
            model, teacher, training_mode=False)
        evaluator = custom_evaluators.DistilConsistencyEvaluator(dev_dataloader_distil_loss,
                                                                 loss_model_distil=distill_loss_validation,
                                                                 loss_model_consistency=teacher_loss_validation)

    model.fit(train_objectives=train_objectives,
              optimizer_params={'lr': params["LEARNING_RATE"]},
              epochs=params["EPOCHS"],
              warmup_steps=params["WARMUP_STEPS"],
              evaluator=evaluator,
              evaluation_steps=params["EVAL_STEPS"],
              output_path=params["SBERT_SAVE_PATH"],
              callback=callback,
              save_best_model=True)


def prepare_for_training(model, params, train_df, val_df):
    if "distill" in params["LOSSES"]:
        similarity_metrics = [
            f'{col}_similarity' for col in params["FEATURES"][2:]]
        train_examples_distil_loss, dev_examples_distil_loss = prepare_data(
            train_df, val_df, similarity_metrics)
    if "distill2" in params["LOSSES"]:
        similarity_metrics = [
            f'{col}_similarity' for col in params["FEATURES"][2:]] + ['similarity']
        train_examples_distil_loss, dev_examples_distil_loss = prepare_data(
            train_df, val_df, similarity_metrics)
    if "consistency" in params["LOSSES"]:
        similarity_metrics = [
            f'{col}_similarity' for col in params["FEATURES"][2:]]
        train_examples_consistency_loss, dev_examples_consistency_loss = prepare_data(
            train_df, val_df, similarity_metrics)
    train_examples_cos_sim_loss, dev_examples_cos_sim_loss = prepare_data(
        train_df, val_df, ['similarity'])

    train_objectives = []

    if "consistency" in params["LOSSES"]:
        # init model for consistency, parameters are frozen
        teacher = SentenceTransformer(params["SBERT_INIT"], device="cuda")
        freeze.freeze_all_layers(teacher)

        train_dataloader_consistency_loss = torch.utils.data.DataLoader(
            train_examples_consistency_loss, shuffle=True, batch_size=params["BATCH_SIZE"])
        consistency_loss = custom_losses.MultipleConsistencyLoss(
            model, teacher)
        train_objectives.append(
            (train_dataloader_consistency_loss, consistency_loss))

    if params["USE_DISTILL"]:
        train_dataloader_distil_loss = torch.utils.data.DataLoader(
            train_examples_distil_loss, shuffle=True, batch_size=params["BATCH_SIZE"])
        dev_dataloader_distil_loss = torch.utils.data.DataLoader(
            dev_examples_distil_loss, shuffle=False, batch_size=params["BATCH_SIZE"])

        loss_model = custom_losses.DistilLoss if "distill" in params[
            "LOSSES"] else custom_losses.DistilLoss2
        distill_loss_training = loss_model(model,
                                           sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                           num_labels=params["N"] + 1,
                                           feature_dim=params["FEATURE_DIM"],
                                           bias_inits=None)
        distill_loss_validation = loss_model(model,
                                             sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                             training_mode=False,
                                             # TODO not sure
                                             num_labels=params["N"] + 1,
                                             feature_dim=params["FEATURE_DIM"],
                                             bias_inits=None)
        train_objectives.append(
            (train_dataloader_distil_loss, distill_loss_training))
        evaluator = custom_evaluators.Distil2Evaluator(
            dev_dataloader_distil_loss, loss_model_distil=distill_loss_validation)

    if "cosine" in params["LOSSES"]:
        train_dataloader_cos_sim_loss = torch.utils.data.DataLoader(
            train_examples_cos_sim_loss, shuffle=True, batch_size=params["BATCH_SIZE"])
        dev_dataloader_cos_sim_loss = torch.utils.data.DataLoader(
            dev_examples_cos_sim_loss, shuffle=False, batch_size=params["BATCH_SIZE"])
        cos_sim_loss = losses.CosineSimilarityLoss(model=model)
        train_objectives.append((train_dataloader_cos_sim_loss, cos_sim_loss))
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            val_df['snippet1'].tolist(), val_df['snippet2'].tolist(), val_df['label'].tolist())

    if "consistency" in params["LOSSES"] and params["USE_DISTILL"]:
        teacher_loss_validation = custom_losses.MultipleConsistencyLoss(
            model, teacher, training_mode=False)
        evaluator = custom_evaluators.DistilConsistencyEvaluator(dev_dataloader_distil_loss,
                                                                 loss_model_distil=distill_loss_validation,
                                                                 loss_model_consistency=teacher_loss_validation)

    return train_objectives, evaluator


# TODO : might need it
# def train(model, params, train_df, val_df):
#     train_objectives, evaluator = prepare_for_training(
#         model, params, train_df, val_df)

#     model_fit = model.fit(train_objectives=train_objectives,
#                           optimizer_params={'lr': params["LEARNING_RATE"]},
#                           epochs=params["EPOCHS"],
#                           warmup_steps=params["WARMUP_STEPS"],
#                           evaluator=evaluator,
#                           evaluation_steps=100,
#                           output_path=params["SBERT_SAVE_PATH"],
#                           callback=callback,
#                           save_best_model=True)
#     return model_fit, evaluator, model


def infer(df, model_path, params):
    print('Loading model from', model_path)
    model = SentenceTransformer(model_path, device=device)

    # example sentence pairs
    xsent = df['snippet1'].tolist()
    ysent = df['snippet2'].tolist()

    # encode with s3bert
    # list 100 x nump.ndarray of shape (384,)
    xsent_encoded = model.encode(xsent)
    ysent_encoded = model.encode(ysent)

    # get similarity scores of different features
    preds = ph.get_preds(xsent_encoded, ysent_encoded,
                         biases=None, n=params["N"], dim=params["FEATURE_DIM"])

    column_names = ['global_pred_similarity']

    if params["USE_DISTILL"]:
        for feature in params["FEATURES"][2:]:
            column_names.append(f'{feature}_pred_similarity')
        column_names.append('residual_pred_similarity')

    pred_df = pd.DataFrame(preds, columns=column_names)
    df_wpred = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    return df_wpred


def get_mse_scores(test_df_wpred, params):
    # pred_column = "residual_pred_similarity" if params["USE_DISTILL"] else "global_pred_similarity"
    # sr_1 = mse(test_df_wpred["label"].tolist(),
    #            test_df_wpred[pred_column].tolist())
    # sr_2 = mse(test_df_wpred["label"].tolist(),
    #            test_df_wpred["similarity"].tolist())
    # return sr_1, sr_2
    mse_global = mse(
        test_df_wpred["label"].tolist(), test_df_wpred["global_pred_similarity"].tolist())
    mse_baseline = mse(
        test_df_wpred["label"].tolist(), test_df_wpred["similarity"].tolist())
    print("Baseline similarity", mse_baseline)
    print("Global similarity", mse_global)
    if params["USE_DISTILL"]:
        mse_residual = mse(
            test_df_wpred["label"].tolist(), test_df_wpred["residual_pred_similarity"].tolist())
        print("Residual similarity", mse_residual)
        for feature in params["FEATURES"][2:]:
            mse_feature = mse(test_df_wpred[f'{feature}_similarity'].tolist(
            ), test_df_wpred[f'{feature}_pred_similarity'].tolist())
            print(f"{feature} similarity", mse_feature)


def get_spearman_scores(test_df_wpred, params):
    spearman_global = spearmanr(
        test_df_wpred["label"].tolist(), test_df_wpred["global_pred_similarity"].tolist())[0]
    spearman_baseline = spearmanr(
        test_df_wpred["label"].tolist(), test_df_wpred["similarity"].tolist())[0]
    print("Baseline similarity", spearman_baseline)
    print("Global similarity", spearman_global)
    if params["USE_DISTILL"]:
        spearman_residual = spearmanr(
            test_df_wpred["label"].tolist(), test_df_wpred["residual_pred_similarity"].tolist())[0]
        print("Residual similarity", spearman_residual)
        for feature in params["FEATURES"][2:]:
            spearman_feature = spearmanr(test_df_wpred[f'{feature}_similarity'].tolist(
            ), test_df_wpred[f'{feature}_pred_similarity'].tolist())[0]
            print(f"{feature} similarity", spearman_feature)

    # train_dataloader, loss, dev_evaluator, test_evaluator, sbert_model = prepare_for_classification_training(
        # classified_df, classifications, params, dataset_dir, batch_size=batch_size, save_model=save_model


def train_combined(classification_train_objectives, similarity_train_objectives, similarity_evaluator, classification_evaluator, params, sbert_model):
    evaluator = custom_evaluators.CombinedEvaluator(
        similarity_evaluator, classification_evaluator)

    train_objectives = classification_train_objectives + similarity_train_objectives
    model_fit = sbert_model.fit(
        train_objectives=train_objectives,
        optimizer_params={'lr': params["LEARNING_RATE"]},
        evaluator=evaluator,
        epochs=params["EPOCHS"],
        evaluation_steps=params["EVALUATION_STEPS"],
        warmup_steps=params["WARMUP_STEPS"],
        output_path=params["SAVE_PATH"],
        callback=callback
    )
    return model_fit, sbert_model
