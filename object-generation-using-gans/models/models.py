
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'glcic_gan':
        raise NotImplementedError("Not Yet there.!")
    
    elif opt.model == 'pix2pixHD':
        assert(opt.dataset_mode == 'aligned')
        # from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        from .pix2pix_model import Pix2PixModel
        print("################################################## Pix2PixHD #############################################")
        model = Pix2PixModel()
        # if opt.isTrain or opt.model == 'train':
        #     model = Pix2PixHDModel()
        # else:
        #     model = InferenceModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    return model
