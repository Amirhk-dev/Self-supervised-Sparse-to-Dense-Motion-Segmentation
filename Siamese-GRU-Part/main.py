from def_functions import *

def main(args):
    device_number = int(args['device_number']) 

    set_the_device(device_number)
    trainvalPairsSet, trainvalTrajsSet = load_train_val_set()

    # datahalf: for each sequence take the first part as train and the second part as val set
    seq_pairs_train = get_trainvalset_for_model(trainvalPairsSet, trainvalTrajsSet, datahalf=0)
    seq_pairs_val   = get_trainvalset_for_model(trainvalPairsSet, trainvalTrajsSet, datahalf=1) 
    
    train_pairs_group = seq_pairs_train     
    val_pairs_group   = seq_pairs_val       

    model = motion_GRU()
    model = model.cuda()
    learning_rate = float(args['learning_rate'])
    optimizer     = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    # train the model
    train_the_model(model, optimizer, train_pairs_group, trainvalTrajsSet, val_pairs_group, args)
    
    weight_list = []
    grad_list = []
    for name in model.named_parameters():
        weight_list.append(name)
        grad_list.append(name[1].grad)
    
    PATH = '../output/gru_model_' + '.pth'
    torch.save(model.state_dict(), PATH)

    # produce prediction for all pairs in the graph
    produce_costs_for_graph(model, seq_pairs_train, args)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i2", "--batch_size", required=True, help="batch size")
    ap.add_argument("-i3", "--number_of_epochs", required=True, help="number of epochs")
    ap.add_argument("-i4", "--learning_rate", required=True, help="learning rate")
    ap.add_argument("-i5", "--device_number", required=True, help="GPU device number")   
    ap.add_argument("-i6", "--number_of_workers", required=True, help="number of workers")   
    args = vars(ap.parse_args())

    main(args)

