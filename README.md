# 程序入口

    main.py

## dataset
集成了datset类和collate_fn算法，每次返回

    src = [src len, batch size, 3]      # grid_x, grid_y, tid（第几个时间步）
    src_gps = [src len, batch size, 2]  # lat, lng
    src_len = tuple(int)[batch size]    # len of src, 包括头部的token
    trg_gps = [trg len, batch size, 2]  # lat, lng
    trg_eid = [trg len, batch size]     # road segment id
    trg_rate = [trg len, batch size]    # moving ratio
    trg_len = tuple(int)[batch size]    # len of trg, 包括头部的token

collate_fn算法对src, src_gps, trg_gps, trg_eid, trg_rate进行pad_sequence处理

## model

encoder, decoder, multi-task

    encoder输入src，src_len，返回hidden

    decoder迭代更新hidden，预测eid

    multi-task以hidden和预测的eid为输入，预测rate

## train

    loss_eid, NLLLoss
    loss_rate, MSELoss

## data

    edge_box: 某条道路500米内的道路集合
    edge_geometry: {eid:shapely.LingString}
    G_sz2nd: networkx, 包括了边和点信息
    traj_final: 128 * 10 条轨迹

## MTrajRec-main

    原始文献中的代码
