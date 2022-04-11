# coding: utf-8
import io
import sys
import numpy
from tqdm import tqdm
import pickle
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
import torch
from src.expressions_transfer import *
import os


batch_size = 64
embedding_size = 512
hidden_size = 512

n_epochs = 120
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
cnn_channel = 64
gru_layers = 2
tfm_layers = 4



data = load_raw_data("data/math23k_train.json")
pairs, generate_nums, copy_nums = transfer_num(data)
temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_trained = temp_pairs

data = load_raw_data("data/math23k_test.json")
pairs, _, _ = transfer_num(data)
temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_tested = temp_pairs

input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)
print(output_lang.index2word)
with open('models/input_lang.pkl', 'wb') as f:
    pickle.dump(input_lang, f)

with open('models/output_lang.pkl', 'wb') as f:
    pickle.dump(output_lang, f)

# Initialize models
encoder = EncoderGRUTFM(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                        cnn_channel=cnn_channel, gru_layers=gru_layers, tfm_layers=tfm_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True, threshold=0.1, threshold_mode='rel')
predict_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(predict_optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True, threshold=0.1, threshold_mode='rel')
generate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(generate_optimizer, mode='min', factor=0.5, patience=5,
                                                                verbose=True, threshold=0.1, threshold_mode='rel')
merge_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(merge_optimizer, mode='min', factor=0.5, patience=5,
                                                             verbose=True, threshold=0.1, threshold_mode='rel')

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

best_vacc = -1
best_equacc = -1
tt = 0

for epoch in range(n_epochs):
    print('learning rate:', encoder_optimizer.state_dict()['param_groups'][0]['lr'])
    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(
        train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()
    for idx in tqdm(range(len(input_lengths))):
        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang,
            num_pos_batches[idx])
        loss_total += loss

    encoder_scheduler.step(loss_total / len(input_lengths))
    predict_scheduler.step(loss_total / len(input_lengths))
    generate_scheduler.step(loss_total / len(input_lengths))
    merge_scheduler.step(loss_total / len(input_lengths))

    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    if epoch % 10 == 0 or epoch > n_epochs - 5:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
            test_res, _ = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                        merge, output_lang, test_batch[5], beam_size=beam_size)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4],
                                                              test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")

        if best_vacc < value_ac:
            best_vacc = value_ac
            best_equacc = equation_ac
            tt = eval_total
            if (float(value_ac) / eval_total) >= 0.775:
                print("------------------------------------Saving-----------------------------------------")
                torch.save(encoder.state_dict(), "models/encoder_dcte")
                torch.save(predict.state_dict(), "models/predict_dcte")
                torch.save(generate.state_dict(), "models/generate_dcte")
                torch.save(merge.state_dict(), "models/merge_dcte")

print(__file__)
print("***********************************************************************************")
print(best_equacc, best_vacc, tt)
print("best_answer_acc", float(best_equacc) / tt, float(best_vacc) / tt)
print("***********************************************************************************")
