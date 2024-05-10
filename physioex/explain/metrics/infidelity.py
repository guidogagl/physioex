import torch

from typing import Callable

def _infidelity(model : torch.nn.Module,
                input : torch.Tensor, 
                exp : torch.Tensor, 
                label : int , 
                pdt, 
                binary_I, 
                perturb_func : Callable
                ):

    if pert == 'Gaussian':
        image_copy_ind = np.apply_along_axis(set_zero_infid, 1, image_copy, total, point, pert)
    elif pert == 'Square':
        image_copy, ind = get_imageset(image_copy, im_size[1:], rads=rads)

    if pert == 'Gaussian' and not binary_I:
        image_copy = image_copy_ind[:, :total]
        ind = image_copy_ind[:, total:total+point]
        rand = image_copy_ind[:, total+point:total+2*point]
        exp_sum = np.sum(rand*np.apply_along_axis(get_exp, 1, ind, exp_copy), axis=1)
        ks = np.ones(num)
    elif pert == 'Square' and binary_I:
        exp_sum = np.sum(ind * np.expand_dims(exp_copy, 0), axis=1)
        ks = np.apply_along_axis(shap_kernel, 1, ind, X=image.reshape(-1))
        ks = np.ones(num)
    else:
        raise ValueError("Perturbation type and binary_I do not match.")

    image_copy = np.reshape(image_copy, (num, 1, 28, 28))
    image_v = Variable(torch.from_numpy(image_copy.astype(np.float32)).cuda(), requires_grad=False)
    out = forward_batch(model, image_v, FORWARD_BZ)
    pdt_rm = (out[:, label])
    pdt_diff = pdt - pdt_rm

    # performs optimal scaling for each explanation before calculating the infidelity score
    beta = np.mean(ks*pdt_diff*exp_sum) / np.mean(ks*exp_sum*exp_sum)
    exp_sum *= beta
    infid = np.mean(ks*np.square(pdt_diff-exp_sum)) / np.mean(ks)
    return infid


def get_explanation_pdt(image, model, label, exp, sg_r=None, sg_N=None, given_expl=None, binary_I=False):
    image_v = Variable(image, requires_grad=True)
    model.zero_grad()
    out = model(image_v)
    pdtr = out[:, label]
    pdt = torch.sum(out[:, label])

    if exp == 'Grad':
        pdt.backward()
        grad = image_v.grad
        expl = grad.data.cpu().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'GBP':
        gb_model = GuidedBackpropReLUModel(model=copy.deepcopy(model), use_cuda=True)
        gb = gb_model(image_v, index=label)
        expl = gb
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'SHAP':
        expl = shap(image.cpu(), label, pdt, model, 20000)
    elif exp == 'Square':
        expl = optimal_square(image.cpu(), label, pdt, model, 20000)
    elif exp == 'NB':
        expl = optimal_nb(image.cpu(), label, pdt, model, 20000)
        if binary_I:
            expl = expl * image.cpu().numpy().flatten()
    elif exp == 'Int_Grad':
        for i in range(10):
            image_v = Variable(image * i/10, requires_grad=True)
            model.zero_grad()
            out = model(image_v)
            pdt = torch.sum(out[:, label])
            pdt.backward()
            grad = image_v.grad
            if i == 0:
                expl = grad.data.cpu().numpy() / 10
            else:
                expl += grad.data.cpu().numpy() / 10
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'Smooth_Grad':
        avg_points = 50
        for count in range(int(sg_N/avg_points)):
            sample = torch.FloatTensor(sample_eps_Inf(image.cpu().numpy(), sg_r, avg_points)).cuda()
            X_noisy = image.repeat(avg_points, 1, 1, 1) + sample
            expl_eps, _ = get_explanation_pdt(X_noisy, model, label, given_expl, binary_I=binary_I)
            if count == 0:
                expl = expl_eps.reshape(avg_points, int(expl_eps.shape[0]/avg_points),
                                        expl_eps.shape[1], expl_eps.shape[2], expl_eps.shape[3])
            else:
                expl = np.concatenate([expl,
                                       expl_eps.reshape(avg_points, int(expl_eps.shape[0]/avg_points),
                                                        expl_eps.shape[1], expl_eps.shape[2], expl_eps.shape[3])],
                                      axis=0)
        expl = np.mean(expl, 0)
    else:
        raise NotImplementedError('Explanation method not supported.')

    return expl, pdtr