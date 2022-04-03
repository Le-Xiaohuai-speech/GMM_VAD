# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:10:20 2022

The python implementation of WebRTC-vad
@author: xiaohuai.le
"""

import numpy as np
import os
import librosa
import yaml

'''
GMM-base VAD algorithm, steps:
1、分解子带，计算子带对数能量作为特征 f_i
2、对f_i, 使用2高斯的一维高斯分布建模语音和噪声， Gn = w_n1 * Gn1(u_n1,std_n1) + w_n2 * Gn2(u_n2,std_n2) 
                                                Gs = w_s1 * Gn1(u_s1,std_s1) + w_s2 * Gs2(u_s2,std_s2) 
3、判决使用子带似然比阈值
4、语音模型的更新只在语音处，噪声模型的更新只在噪声
'''
class signal_processing():
    
    def __init__(self, frame_length, fs):
        self.N = frame_length // 2
        if fs == 8000:
            self.index = [round(0.02*self.N), round(0.0625*self.N), round(0.125*self.N), 
                          round(0.25*self.N), round(0.5*self.N), round(0.75*self.N), self.N]
        elif fs == 16000:
            self.index = [round(0.01*self.N), round(0.03125*self.N), round(0.0625*self.N), 
                          round(0.125*self.N), round(0.25*self.N), round(0.375*self.N), self.N]
            
    def sub_band_energy(self, signal):
        '''
        calculate the logarithmic energy of 6 subbands
        Instead of the band-pass filters, I use the FFT to get sub-bands
        80-250,250-500,500-1000,1000-2000,2000-3000,3000-4000 Hz
        '''
        sub_energy = np.zeros(6, dtype = 'float32')
        f_amp = abs(np.fft.rfft(signal)).astype('float32')
        
        sub_energy[0] = np.sum(f_amp[self.index[0]:self.index[1]]**2)  
        sub_energy[1] = np.sum(f_amp[self.index[1]:self.index[2]]**2) 
        sub_energy[2] = np.sum(f_amp[self.index[2]:self.index[3]]**2) 
        sub_energy[3] = np.sum(f_amp[self.index[3]:self.index[4]]**2) 
        sub_energy[4] = np.sum(f_amp[self.index[4]:self.index[5]]**2) 
        sub_energy[5] = np.sum(f_amp[self.index[5]:self.index[6]]**2) 
        
        total_energy = 10*np.log10(np.sum(sub_energy) / N +1e-6)
        
        return 10 * np.log10( sub_energy / self.N + 1e-6), total_energy
    
    @staticmethod
    def cal_Gaussian_prob(x,miu,sigma):
        '''
        calculate the probability of a Gaussian
        temp2 and temp3 are used for updating
        '''
        temp = 1 / sigma
        temp2 = (x-miu)*temp**2  
        temp1 = (x-miu)*temp2
        temp3 = temp*(temp1- 1 )
        return temp*np.exp(-0.5*temp1),temp2,temp3
    
    def log_likely_ratio(self, x,n_mean,n_std,n_weights,s_mean,s_std,s_weights):
        '''
        calculate the logarithmic likelihood ratio of speech model and noise model
        x: the logarithmic of a subband
        n_mean, n_std, n_weights: the parameters of the noise GMM of a subband
        s_mean, s_std, s_weights：the parameters of the speech GMM of a subband
        
        the log-likelihood ratio is calculated as:
        H1 = w_s1 * N(miu_s1,sigma_s1) + w_s2 * N(miu_s2,sigma_s2)
        H0 = w_n1 * N(miu_n1,sigma_n1) + w_n2 * N(miu_n2,sigma_n2) 
        p = log(H1/H0)
          = log(H1) - log(H0)
        '''
        probs0, deltas_0_m , deltas_0_std = self.cal_Gaussian_prob(x, s_mean[0], s_std[0])
        probs1, deltas_1_m , deltas_1_std = self.cal_Gaussian_prob(x, s_mean[1], s_std[1])
        probn0, deltan_0_m , deltan_0_std = self.cal_Gaussian_prob(x, n_mean[0], n_std[0]) 
        probn1, deltan_1_m , deltan_1_std = self.cal_Gaussian_prob(x, n_mean[1], n_std[1])
        
        deltas = np.array([[deltas_0_m,deltas_1_m],[deltas_0_std,deltas_1_std]])
        deltan = np.array([[deltan_0_m,deltan_1_m],[deltan_0_std,deltan_1_std]])
        
        H1 = s_weights[0] * probs0 + s_weights[1] * probs1
        H0 = n_weights[0] * probn0 + n_weights[1] * probn1
        #limitation
        if H1 < 1e-6:
            logh1 = -10
        else: 
            logh1 = np.log(H1)
            
        if H0 < 1e-6:
            logh0 = -10
        else:
            logh0 = np.log(H0)
    
        ratio = logh1 - logh0
    
        return ratio ,H1, H0, deltas, deltan

class VAD_detector(signal_processing):
    
    def __init__(self, config_file = './default_parameters.yaml', fs = 8000, load = 0, fL = 0.02):
        
        self.fL = int(fL * fs) 
        signal_processing.__init__(self, frame_length = self.fL, fs = fs)
        self.config_data = self.read_para(config_file)
        self.fs = fs
        # model parameters
        self.kNoiseDataWeights = np.array(self.config_data['kNoiseDataWeights'],dtype = 'float32')         #噪声模型的权重6*2=12
        self.kSpeechDataWeights = np.array(self.config_data['kSpeechDataWeights'],dtype = 'float32')       #语音模型的权重6*2=12
        self.kNoiseDataMeans = np.array(self.config_data['kNoiseDataMeans'],dtype = 'float32')             #噪声模型的均值6*2=12
        self.kSpeechDataMeans = np.array(self.config_data['kSpeechDataMeans'],dtype = 'float32')           #语音模型的均值6*2=12
        self.kNoiseDataStds = np.array(self.config_data['kNoiseDataStds'],dtype = 'float32')               #噪声模型的标准差6*2=12
        self.kSpeechDataStds = np.array(self.config_data['kSpeechDataStds'],dtype = 'float32')             #语音模型的标准差6*2=12
        # thresholds and limitations
        self.kMinenergy = self.config_data['kMinenergy']
        self.kMinimumDifference = np.array(self.config_data['kMinimumDifference'],dtype = 'float32')       #最小两gmm质心距离
        kMaximumSpeech = self.config_data['kMaximumSpeech']
        kMaximumNoise = self.config_data['kMaximumNoise']
        self.kMaximumSpeech = np.concatenate([kMaximumSpeech, kMaximumSpeech]).astype('float32')  #最大语音均值
        self.kMaximumNoise = np.concatenate([kMaximumNoise, kMaximumNoise]).astype('float32')     #最大噪声均值
        self.kMinimumMean = self.config_data['kMinimumMean']                                      #最小高斯均值
        self.kMinStd = self.config_data['kMinStd']                                                #最小高斯标准差
        self.kLocalThresholdQ = self.config_data['kLocalThresholdQ']                              #子带阈值
        self.kGlobalThresholdQ = self.config_data['kGlobalThresholdQ']                            #子带加权阈值
        self.kSpectrumWeight = self.config_data['kSpectrumWeight']                                #子带似然比权重
        self.kMaxNonespeech = 5                                   #最小语音间隔
        
        #相关的超参数和需要的数组
        self.log_likelihood = np.zeros([6])            #两个模型的似然比
        self.h1 = np.zeros([6])                        #语音模型的似然
        self.h0 = np.zeros([6])                        #高斯模型的似然
        
        #噪声均值队列, 噪声的均值将在过去一百帧内被保存，每次找其中最小的16个
        self.minimum = np.zeros([6,100]).astype('float32') + 1000
        
        #这里存着用于更新均值和方差的系数
        self.S_mean_N = np.zeros([6,2],dtype = 'float32')               #更新语音均值计算的梯度                         
        self.N_mean_N = np.zeros([6,2],dtype = 'float32')               #更新噪声均值计算的梯度
        self.S_std_N = np.zeros([6,2],dtype = 'float32')                #更新语音标准差计算的梯度
        self.N_std_N = np.zeros([6,2],dtype = 'float32')                #更新噪声标准差计算的梯度
        
        #连续噪声帧长度             
        self.Kn = 0.02  #噪声均值更新系数
        self.Ks = 0.2   #语音均值更新系数
        self.Cn = 0.1   #噪声语音标准差更新系数
        self.KL = 0.6   #噪声均值持续更新系数
        self.alpha1 = 0.99
        self.alpha2 = 0.2
        self.median = np.zeros(6)              #这里存着最后输出的最小值的中位数
        self.i = 0
        
    def read_para(self, file):
        f = open(file,'r',encoding='utf-8')
        data = f.read()
        return yaml.load(data)
    
    def cal_ratio(self,signal):
        
        vad_results = 0
        # get energy
        feature, total_energy = self.sub_band_energy(signal)
        sum_log_likelihood = 0
        # global energy threshold
        if total_energy > self.kMinenergy:
            for j in range(6):
                self.log_likelihood[j], _, _, deltas, deltan = self.log_likely_ratio(feature[j],
                                            [self.kNoiseDataMeans[j], self.kNoiseDataMeans[j+6]],
                                            [self.kNoiseDataStds[j], self.kNoiseDataStds[j+6]],
                                            [self.kNoiseDataWeights[j], self.kNoiseDataWeights[j+6]],
                                            [self.kSpeechDataMeans[j], self.kSpeechDataMeans[j+6]],
                                            [self.kSpeechDataStds[j], self.kSpeechDataStds[j+6]],
                                            [self.kSpeechDataWeights[j], self.kSpeechDataWeights[j+6]])
                self.S_mean_N[j,:] = deltas[0,:]
                self.N_mean_N[j,:] = deltan[0,:]
                self.S_std_N[j,:] = deltas[1,:]
                self.N_std_N[j,:] = deltan[1,:]
                # 6是设置的阈值
                if self.log_likelihood[j] > self.kLocalThresholdQ[1] / 6 :
                    vad_results = 1
                
                sum_log_likelihood = self.log_likelihood[j] * self.kSpectrumWeight[j]
            # 10是设置的阈值
            vad_results = vad_results or (sum_log_likelihood / 10 > self.kGlobalThresholdQ[2])  
        #%%
        #gmm模型的更新
        average_noise_means = np.sum(self.kNoiseDataMeans.reshape([2,6]) * self.kNoiseDataWeights.reshape([2,6]),axis = 0)
        #获取最小的16个噪声,并取其中位数作为更新辅助
        self.minimum[:,1:] = self.minimum[:,0:99]
        self.minimum[:,0] = feature
       
        self.median_last = self.median
        if self.i < 16: 
            self.median = np.median(self.minimum[:,:self.i+1],axis = 1)
            if self.i==0:
                self.median_last = self.median
        else:
            #噪声最小值跟随
            #self.median = np.median(np.array([heapq.nsmallest(16,self.minimum[i,:]) for i in range(6)]),axis = 1)
            self.median = np.min(self.minimum,axis = 1)
        for m in range(6):
            #如果噪声上升过快的话，我们将用小幅度系数进行更新，如果下降的话我们就用大系数
            if self.median[m] >= self.median_last[m]:
                self.median[m] = self.alpha1 * self.median_last[m] + (1-self.alpha1) * self.median[m]
            else:
                self.median[m] = self.alpha2 * self.median_last[m] + (1-self.alpha2) * self.median[m]
        #利用gmm的公式计算以下概率用于更新每一个高斯的均值和方差
        # log_likelihood = np.log(hi/h0)
        #p0 = h0/(h0+h1) = 1/(1+exp(log_likelihood))
        #p1 = h1/(h0+h1) = 1/(1+exp(-log_likelihood))
        if not vad_results:
            #更新噪声模型。
            p0 = 1/(1+np.exp(self.log_likelihood[:]))
            #更新均值
            self.kNoiseDataMeans[:6] += self.Kn * self.N_mean_N[:,0] * p0
            self.kNoiseDataMeans[6:] += self.Kn * self.N_mean_N[:,1] * p0
            #限制均值最大值
            self.kNoiseDataMeans[self.kNoiseDataMeans > self.kMaximumNoise] = self.kMaximumNoise[self.kNoiseDataMeans > self.kMaximumNoise]
            #更新方差
            self.kNoiseDataStds[:6] += self.Cn * self.N_std_N[:,0] * p0
            self.kNoiseDataStds[6:] += self.Cn * self.N_std_N[:,1] * p0
            #限制方差最小值
            self.kNoiseDataStds[self.kNoiseDataStds < self.kMinStd] = self.kMinStd
        else:
            #更新语音模型
            p1 = 1/(1+np.exp(-self.log_likelihood[:]))
            #更新均值
            self.kSpeechDataMeans[:6] += self.Ks * self.S_mean_N[:,0] * p1
            self.kSpeechDataMeans[6:] += self.Ks * self.S_mean_N[:,1] * p1
            #限制均值最大值
            self.kSpeechDataMeans[self.kSpeechDataMeans > self.kMaximumSpeech] = self.kMaximumSpeech[self.kSpeechDataMeans > self.kMaximumSpeech]
            #更新方差
            self.kSpeechDataStds[:6] += self.Cn * self.S_std_N[:,0] * p1
            self.kSpeechDataStds[6:] += self.Cn * self.S_std_N[:,0] * p1
            #限制方差最小值
            self.kSpeechDataStds[self.kSpeechDataStds < self.kMinStd] = self.kMinStd
            
        #噪声长期均值更新，与vad标志无关
        delta_n = self.KL * (self.median - average_noise_means)
        self.kNoiseDataMeans[:6] += delta_n
        self.kNoiseDataMeans[6:] += delta_n
        #检测噪声和语音混合高斯均值。如果过于靠近就进行分离，分离到距离kMinimumDifference为止
        average_noise_means = np.sum(self.kNoiseDataMeans.reshape([2,6]) * self.kNoiseDataWeights.reshape([2,6]),axis = 0)
        average_speech_means = np.sum(self.kSpeechDataMeans.reshape([2,6]) * self.kSpeechDataWeights.reshape([2,6]),axis = 0)
        diff = average_speech_means - average_noise_means

        temp = np.zeros(6)
        for i in range(6):
            if diff[i] < self.kMinimumDifference[i]:
                temp[i] = self.kMinimumDifference[i] - diff[i]
                
        self.kNoiseDataMeans[:6] += -0.2*temp
        self.kNoiseDataMeans[6:] += -0.2*temp
        self.kSpeechDataMeans[:6] += 0.8*temp
        self.kSpeechDataMeans[6:] += 0.8*temp
        
        self.i += 1
        
        return vad_results
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    v = VAD_detector()
    s = librosa.load('D:/codes/test_audio/clean/440C020a.wav',8000)[0]
    s = np.int16(s*2**15)+np.random.randn(len(s))*100
    H = 160
    N = len(s) // H
    sp = signal_processing(H,8000)
    feature = np.zeros([6,N],dtype = 'float32')
    vad = np.zeros(N)
    # real-time vad
    vad_s = np.zeros(N*H)
    for i in range(N):
        feature[:,i] = sp.sub_band_energy(s[i*H:(i+1)*H])[0]
        vad[i] = v.cal_ratio(s[i*H:(i+1)*H])
        if vad[i]:
            vad_s[i*H:(i+1)*H] = 1
    
    plt.subplot(211)
    plt.plot(s / np.max(abs(s)))
    plt.plot(vad_s)

    plt.subplot(212)
    plt.imshow(feature)
    plt.plot(vad*5,'orange')
    
    