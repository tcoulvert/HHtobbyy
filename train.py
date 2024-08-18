# Common Py packages
import numpy as np

# ML packages
import torch
from torch.autograd import Variable

# Module packages
from EarlyStopping import EarlyStopping

def train(
    num_epochs, model, criterion, optimizer, scheduler, 
    state_filename, model_filename, volatile=False, data_loader=None, save_model=True
):
    best_model = model.state_dict()
    best_acc = 0.0
    train_losses ,val_losses = [],[]
    callback = EarlyStopping(patience=10)
    callback.on_train_begin()
    breakdown = False
    for epoch in range(num_epochs):
        if breakdown:
            print("Early stopped.")
            break
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            running_loss = 0.0
            running_corrects = 0
            if phase == 'training':
                model.train() # Set model to training mode
                volatile=False
            else:
                model.eval() # Set model to evaluate mode
                volatile=True
            

            # Iterate over data.
            for batch_idx, (particles_data, hlf_data, y_data) in enumerate(data_loader[phase]):
                particles_data = particles_data.numpy()
                arr = np.sum(particles_data!=0, axis=1)[:,0] # the number of particles in each batch
                arr = [1 if x==0 else x for x in arr]
                arr = np.array(arr)
                sorted_indices_la = np.argsort(-arr)
                particles_data = torch.from_numpy(particles_data[sorted_indices_la]).float()
                hlf_data = hlf_data[sorted_indices_la]
                y_data = y_data[sorted_indices_la]
                particles_data = Variable(particles_data, requires_grad=not volatile).cuda() 
                # particles_data = Variable(particles_data, requires_grad=not volatile)
                
                hlf_data = Variable(hlf_data, requires_grad=not volatile).cuda()
                # hlf_data = Variable(hlf_data, requires_grad=not volatile)
                y_data = Variable(y_data, requires_grad=False).cuda()
                # y_data = Variable(y_data, requires_grad=not volatile)
                t_seq_length = [arr[i] for i in sorted_indices_la]
                particles_data = torch.nn.utils.rnn.pack_padded_sequence(particles_data, t_seq_length, batch_first=True)
                
                if phase == 'training':
                    optimizer.zero_grad()
                # forward pass
                outputs = model(particles_data, hlf_data)
                _, preds = torch.max(outputs.data, 1)
                # loss = criterion(outputs, y_data)
                print(f"y_data shape = {y_data.shape}")
                print(f"unsqueezed y_data shape = {torch.unsqueeze(y_data, 1).shape}")
                print(f"output shape = {outputs.shape}")
                print(f"squeezed output shape = {torch.squeeze(outputs).shape}")
                # loss = criterion(outputs, torch.unsqueeze(y_data, 1))
                loss = criterion(torch.squeeze(outputs), y_data)
                
                # backward + optimize only if in training phase
                if phase == 'training':
                    loss.backward()
                    optimizer.step()
                
                # statistics
                # running_loss += loss.data[0]
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == y_data.data)
            
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = 100. * running_corrects / len(data_loader[phase].dataset)
            if phase == 'training':
                train_losses.append(epoch_loss)
            else:
                scheduler.step(epoch_loss)
                val_losses.append(epoch_loss)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                print('Saving..')
                state = {
                    'net': model, #.module if use_cuda else net,
                    'epoch': epoch,
                    'best_acc':epoch_acc,
                    'train_loss':train_losses,
                    'val_loss':val_losses,
                }
                if save_model == True:
                    torch.save(state, state_filename)
                    torch.save(model.state_dict(), model_filename)
                best_acc = epoch_acc
                best_model = model.state_dict()
            if phase == 'validation':
                # breakdown = callback.on_epoch_end(epoch, -epoch_acc)
                breakdown = callback.on_epoch_end(epoch, epoch_loss)
                
         
    print('Best val acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model)
    print('-' * 10)
    return best_acc, train_losses, val_losses
