import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import soundfile as sf
# load phonemizer
import phonemizer
import random
random.seed(0)

import numpy as np
np.random.seed(0)
# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# from inflector import convert_currency

textclenaer = TextCleaner()

# numbers = [164, 169, 174, 179, 184, 189, 194, 199]
numbers = [149]
model_name = 'ClarrisaUn-V1'
for number in numbers:
    reference_audio =  "Models/clarrisa-V1/reference.wav"
    if len(str(number)) == 2:
        audio_model = f"Models/{model_name}/epoch_2nd_000{number}.pth"
    else:
        audio_model = f"Models/{model_name}/epoch_2nd_00{number}.pth"
    config_file = f"Models/{model_name}/config_ft.yml"
    # config_file = "Models/AnaCastano-v1/config_ft.yml"

    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mean, std = -4, 4

    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def preprocess(wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor

    def compute_style(path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

    config = yaml.safe_load(open(config_file))

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]


    def load_model(model_name):
        # params_whole = torch.load("Models/LJSpeech/epoch_2nd_00144.pth", map_location='cpu')
        params_whole = torch.load(model_name, map_location='cpu')
        return params_whole

    params_whole = load_model(audio_model)
    params = params_whole['net']

    for key in model:
        if key in params:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    #             except:
    #                 _load(params[key], model[key])
    _ = [model[key].eval() for key in model]



    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )

    def inference(text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=3, embedding_scale=1):
        text = text.strip()
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

            s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                num_steps=diffusion_steps).squeeze(1)


            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = model.predictor.text_encoder(d_en, 
                                            s, input_lengths, text_mask)

            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = model.decoder(asr, 
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
        squeezed_out = out.squeeze()
    
        # Calculate the number of samples to remove based on the percentage
        remove_percentage = 0.77790742901 / 100
        num_samples_to_remove = int(squeezed_out.shape[-1] * remove_percentage)
        
        # Remove the last 0.46468401487% of samples
        cropped_out = squeezed_out[..., :-num_samples_to_remove]

        return cropped_out.cpu().numpy()
        print(out.shape)
        print(out.squeeze().shape)
        squeezed_out = out.squeeze()
        print(squeezed_out.shape[-1])
        cropped_out = squeezed_out[..., :-50]
        print(cropped_out.shape)

        return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

    def infer(text, outname, speed_up):
        # text = convert_currency(text)
        def split_text(text, max_words=30):
            words = text.split()
            if len(words) <= max_words:
                return [text]

            chunks = []
            start_idx = 0

            while start_idx < len(words):
                end_idx = min(start_idx + max_words, len(words))
                chunk = ' '.join(words[start_idx:end_idx])

                if end_idx < len(words):
                    last_punctuation = max(chunk.rfind('.'), chunk.rfind(','), chunk.rfind(';'))
                    if last_punctuation != -1:
                        end_idx = start_idx + chunk[:last_punctuation + 1].count(' ') + 1
                        chunk = ' '.join(words[start_idx:end_idx])
                
                print(chunk.strip()) # Print each chunk
                if len(chunk.split(' ')) < 5:
                    chunks[-1] = chunks[-1] + ' ' + chunk.strip()
                else:
                    chunks.append(chunk.strip())
                start_idx = end_idx

            return chunks



        def synthesize_and_compute_rtf(text, ref_s, speed_up):
            start_gen = time.time()
            wav = inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=3, embedding_scale=1)
            compute_time = time.time() - start_gen
            print('Compute style time:', compute_time)
            
            if speed_up != 1:  # Only apply time-stretching if speed_up is not 1
                fast_wav = librosa.effects.time_stretch(wav, rate = speed_up)
            else:
                fast_wav = wav
            rtf = (time.time() - start) / (len(wav) / 24000)
            print(f"RTF = {rtf:5f}")
            return fast_wav, compute_time

        reference_dicts = {'696_92939': reference_audio}
        start = time.time()
        noise = torch.randn(1,1,256).to(device)

        for k, path in reference_dicts.items():
            ref_s = compute_style(path)

            text_chunks = split_text(text)
            synthesized_wavs = []

            for chunk in text_chunks:
                wav, compute_time = synthesize_and_compute_rtf(chunk, ref_s, speed_up)
                synthesized_wavs.append(wav)

            wav = np.concatenate(synthesized_wavs)

            synthesized_filename = f"{outname}"
            sf.write(synthesized_filename, wav.squeeze(), 24000)
            print(f"{k} Synthesized: Saved to {synthesized_filename}")

            end_gen = time.time()
            print(str(end_gen - start))

        return outname, str(end_gen - start)
    

    # def infer(text, outname):
    #     reference_dicts = {}
    #     reference_dicts['696_92939'] = reference_audio

    #     start = time.time()
    #     noise = torch.randn(1,1,256).to(device)

    #     for k, path in reference_dicts.items():
    #         start_gen = time.time()
    #         ref_s = compute_style(path)
    #         compute_time = time.time() - start_gen
    #         print('Compute style time: ', compute_time)
    #         # Synthesize the waveform
    #         wav = inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=5)
            
    #         # Calculate real-time factor
    #         rtf = (time.time() - start) / (len(wav) / 24000)
    #         print(f"RTF = {rtf:5f}")
            
    #         # Save synthesized audio
    #         synthesized_filename = f"{outname}"
    #         sf.write(synthesized_filename, wav.squeeze(), 24000)  # Assuming 'wav' is already a numpy array
    #         print(f"{k} Synthesized: Saved to {synthesized_filename}")
    #         end_gen = time.time()
    #         print(str(end_gen-start_gen))
    #     return outname, str(end_gen-start_gen)

    salesperson_texts = [
      {"filename": f"{number}_1.wav", "text": "Hello Valeria! Let me introduce you to our groundbreaking product, the Verizon 5G Home Internet. This service is designed to revolutionize the way you connect to the internet at home, providing ultra fast and reliable internet speeds that can handle all your streaming, gaming, and browsing needs without a hitch."},
    {"filename": f"{number}_2.wav", "text": "For instance, our iPads come with advanced analytics, providing real time shopping tips and enhancing customer engagement. You'll also get high speed 5 G internet, which is faster and more reliable than 4 G. Plus, our plan includes seamless integration with your existing POS systems, eliminating the need for additional software or hardware investments."},
    {"filename": f"{number}_3.wav", "text": "Hi there.... I'm doing great, thanks for asking.... I'm excited to catch up with you and discuss how Verizon Business Unlimited plans can benefit your wearup shoes business. How about you?"},
    {"filename": f"{number}_4.wav", "text": "Verizon 5G Home Internet is also built for the future. As more devices in your home become smart and connected, our network will provide the necessary speed and reliability to support them all. From smart thermostats and security systems to voice assistants and beyond, you can trust Verizon to keep your smart home running seamlessly."},
    {"filename": f"{number}_5.wav", "text": "We're offering amazing deals for new customers, including a free month of service and discounts on our cutting edge 5G compatible devices. Plus, with Verizon's commitment to customer service, you can count on us to provide support whenever you need it."},
    {"filename": f"{number}_6.wav", "text": "Don't miss out on the future of home internet. Experience the power and convenience of Verizon 5G Home Internet today. Visit our website or your nearest Verizon store to learn more and sign up. With Verizon, you're always connected to what matters most."},
    {"filename": f"{number}_7.wav", "text": "With our 5G Home Internet, enjoy seamless connectivity across multiple devices, ensuring that everyone in your household can stream, work, and play without interruptions."},
    {"filename": f"{number}_8.wav", "text": "Take advantage of our robust customer support and cutting edge technology to stay ahead in today's fast paced digital world. Our team is here to assist you 24/7 with any questions or issues you may have."},
    {"filename": f"{number}_9.wav", "text": "Our network is designed to provide superior coverage, even in areas where traditional internet services struggle. Experience the difference with Verizon's unparalleled 5G technology."},
    {"filename": f"{number}_10.wav", "text": "With our 5G Home Internet, you can effortlessly connect your smart home devices, ensuring they work together seamlessly for an enhanced living experience. Stay connected and in control, all the time."},
    {"filename": f"{number}_11.wav", "text": "Enjoy the freedom of high-speed internet without any hidden fees or surprises. Our transparent pricing and flexible plans are designed to fit your needs and budget."},
    {"filename": f"{number}_12.wav", "text": 'It is very exciting for me to teach a student who is so much interested in learning, and do you have any other questionjane?'},]


#     salesperson_texts = [
#     {"filename": f"{number}_1.wav", "text": "¡Hola Valeria! Permíteme presentarte nuestro revolucionario producto, el Verizon 5G Home Internet. Este servicio está diseñado para revolucionar la forma en que te conectas a internet en casa, proporcionando velocidades de internet ultra rápidas y fiables que pueden manejar todas tus necesidades de streaming, juegos y navegación sin problemas."},
#     {"filename": f"{number}_2.wav", "text": "Por ejemplo, nuestros iPads vienen con analíticas avanzadas, proporcionando consejos de compras en tiempo real y mejorando la participación del cliente. También obtendrás internet 5G de alta velocidad, que es más rápido y fiable que el 4G. Además, nuestro plan incluye una integración perfecta con tus sistemas POS existentes, eliminando la necesidad de inversiones adicionales en software o hardware."},
#     {"filename": f"{number}_3.wav", "text": "Hola.... Estoy muy bien, gracias por preguntar.... Estoy emocionado de ponernos al día y discutir cómo los planes ilimitados de negocios de Verizon pueden beneficiar a tu negocio de calzado Wearup. ¿Y tú?"},
#     {"filename": f"{number}_4.wav", "text": "El Verizon 5G Home Internet también está preparado para el futuro. A medida que más dispositivos en tu hogar se vuelvan inteligentes y conectados, nuestra red proporcionará la velocidad y fiabilidad necesarias para soportarlos todos. Desde termostatos inteligentes y sistemas de seguridad hasta asistentes de voz y más allá, puedes confiar en Verizon para mantener tu hogar inteligente funcionando sin problemas."},
#     {"filename": f"{number}_5.wav", "text": "Ofrecemos increíbles ofertas para nuevos clientes, incluyendo un mes de servicio gratis y descuentos en nuestros dispositivos compatibles con 5G de última generación. Además, con el compromiso de servicio al cliente de Verizon, puedes contar con nosotros para proporcionar soporte siempre que lo necesites."},
#     {"filename": f"{number}_6.wav", "text": "No te pierdas el futuro del internet en casa. Experimenta el poder y la conveniencia del Verizon 5G Home Internet hoy. Visita nuestro sitio web o tu tienda Verizon más cercana para más información y para registrarte. Con Verizon, siempre estás conectado a lo que más importa."},
#     {"filename": f"{number}_7.wav", "text": "Con nuestro Internet Hogar 5G, disfruta de una conectividad ininterrumpida en múltiples dispositivos, asegurando que todos en tu hogar puedan transmitir, trabajar y jugar sin interrupciones."},
#     {"filename": f"{number}_8.wav", "text": "Aprovecha nuestro robusto soporte al cliente y la tecnología de vanguardia para mantenerte adelante en el mundo digital de hoy. Nuestro equipo está aquí para asistirte las 24 horas del día, los 7 días de la semana con cualquier pregunta o problema que puedas tener."},
#     {"filename": f"{number}_9.wav", "text": "Nuestra red está diseñada para proporcionar una cobertura superior, incluso en áreas donde los servicios de internet tradicionales tienen dificultades. Experimenta la diferencia con la tecnología 5G sin paralelo de Verizon."},
#     {"filename": f"{number}_10.wav", "text": "Con nuestro Internet Hogar 5G, puedes conectar sin esfuerzo tus dispositivos inteligentes en casa, asegurando que trabajen juntos de manera fluida para una experiencia de vida mejorada. Mantente conectado y en control, todo el tiempo."}
# ]




    for info_dict in salesperson_texts:
        text = info_dict['text']
        outname = info_dict['filename']
        infer(text,outname,1)