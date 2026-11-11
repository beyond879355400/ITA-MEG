import unittest
import numpy as np
from supervisor_metrics import metrics,constant_baselines,supported_aus,qualify


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.y=np.tile(np.arange(6,dtype=float)[:,None],(10,12))
        self.subs=['a']*30+['b']*30
    def test_perfect(self):
        m=metrics(self.y,self.y,self.subs)
        self.assertEqual(m['macro_mae'],0)
        self.assertAlmostEqual(m['per_au'][0]['pearson'],1)
        self.assertEqual(m['per_au'][0]['active_n'],50)
    def test_active_is_equal_au_not_pooled(self):
        y=np.zeros((20,12));y[0,:]=2;y[:,0]=2
        p=y.copy();p[:,0]=1;p[0,1:]=0
        m=metrics(y,p)
        self.assertAlmostEqual(m['active_mae'],23/12)
    def test_constant_fails_correlations(self):
        b=constant_baselines(self.y,self.y,self.subs)
        m=b['train_mean']['metrics'];self.assertIsNone(m['per_au'][0]['pearson'])
        self.assertFalse(qualify(m,b,supported_aus(self.y))['passed'])
    def test_perfect_qualifies(self):
        b=constant_baselines(self.y,self.y,self.subs)
        self.assertTrue(qualify(metrics(self.y,self.y),b,supported_aus(self.y))['passed'])
    def test_support_labels_only(self):
        y=self.y.copy();y[:,0]=0;y[:,1]=0;y[:9,1]=1
        self.assertEqual(supported_aus(y),list(range(2,12)))
    def test_baseline_train_only(self):
        train=np.ones((4,12));b=constant_baselines(train,self.y,self.subs)
        self.assertEqual(b['train_mean']['constant_raw'],[1]*12)
        self.assertEqual(b['train_median']['constant_raw'],[1]*12)
    def test_bad_shape_or_nan(self):
        with self.assertRaises(ValueError): metrics(self.y,self.y[:2])
        with self.assertRaises(ValueError): metrics(self.y[:0],self.y[:0])
        p=self.y.copy();p[0,0]=np.nan
        with self.assertRaises(ValueError):metrics(self.y,p)
    def test_noop_zero_fails_active(self):
        b=constant_baselines(self.y,self.y,self.subs)
        g=qualify(b['zero']['metrics'],b,supported_aus(self.y))
        self.assertFalse(g['checks']['active_improves_5pct'])
    def test_no_active_is_na_and_not_qualified(self):
        y=np.zeros((20,12));b=constant_baselines(y,y,['a']*20)
        self.assertIsNone(b['zero']['metrics']['active_mae'])
        self.assertFalse(qualify(b['zero']['metrics'],b,[])['passed'])

    def test_epoch_resume_rng_and_optimizer(self):
        import torch,random
        from train_head_only import rng_state,restore_rng
        torch.manual_seed(42);np.random.seed(42);random.seed(42)
        g=torch.Generator().manual_seed(42)
        model=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.Sigmoid())
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        x=torch.randn(12,4);y=torch.randn(12,3)
        def step():
            order=torch.randperm(12,generator=g)
            opt.zero_grad();loss=(model(x[order])-y[order]).square().mean();loss.backward();opt.step()
            return order,loss.item()
        step()
        import copy
        state=copy.deepcopy(model.state_dict());os=copy.deepcopy(opt.state_dict())
        rs=rng_state(torch,np,g,'cpu')
        order1,loss1=step();expected=copy.deepcopy(model.state_dict())
        model.load_state_dict(state);opt.load_state_dict(os);restore_rng(torch,np,g,rs,'cpu')
        order2,loss2=step()
        self.assertTrue(torch.equal(order1,order2));self.assertEqual(loss1,loss2)
        for k,v in expected.items():self.assertTrue(torch.equal(v,model.state_dict()[k]))

    def test_layer4_freeze_and_bn(self):
        import torch
        from train_layer4 import freeze_except_layer4,training_mode_frozen_bn
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.backbone=torch.nn.Module()
                self.backbone.early=torch.nn.Conv2d(3,3,1)
                self.backbone.layer4=torch.nn.Sequential(torch.nn.Conv2d(3,3,1),torch.nn.BatchNorm2d(3))
                self.head=torch.nn.Linear(3,2)
            def forward(self,x):return self.head(self.backbone.layer4(self.backbone.early(x)).mean((2,3)))
        m=Tiny();freeze_except_layer4(m);training_mode_frozen_bn(m,torch)
        self.assertFalse(m.backbone.early.weight.requires_grad)
        self.assertTrue(m.backbone.layer4[0].weight.requires_grad)
        self.assertFalse(m.backbone.layer4[1].training)
        mean=m.backbone.layer4[1].running_mean.clone();variance=m.backbone.layer4[1].running_var.clone()
        m(torch.randn(4,3,8,8)).sum().backward()
        self.assertIsNone(m.backbone.early.weight.grad)
        self.assertIsNotNone(m.backbone.layer4[0].weight.grad)
        self.assertTrue(torch.equal(mean,m.backbone.layer4[1].running_mean))
        self.assertTrue(torch.equal(variance,m.backbone.layer4[1].running_var))

if __name__=='__main__': unittest.main()
