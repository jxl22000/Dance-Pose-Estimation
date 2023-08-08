import math
from moviepy.editor import *
import librosa
import matplotlib.pyplot as plt


def audio(filename1, filename2, time):

    clip1 = AudioFileClip(filename1)
    print(clip1.duration)
    if clip1.duration > time:
        clip1 = clip1.subclip(0, time)
    clip1.write_audiofile(filename1 + '.wav')

    clip2 = AudioFileClip(filename2)
    print(clip2.duration)
    if clip2.duration > time * 2:
        clip2 = clip2.subclip(0, time * 2)
    clip2.write_audiofile(filename2 + '.wav')

    return clip1.filename, clip2.filename


def matchClips(filename1, filename2):

    hop_length = 1024
    y_ref, sr1 = librosa.load(filename1 + '.wav')
    y_comp, sr2 = librosa.load(filename2 + '.wav')

    librosa.util.normalize(y_ref)
    librosa.util.normalize(y_comp)

    chroma_ref = librosa.feature.chroma_cqt(y=y_ref, sr=sr1, hop_length=hop_length)
    chroma_comp = librosa.feature.chroma_cqt(y=y_comp, sr=sr2, hop_length=hop_length)

    # Use time-delay embedding to get a cleaner recurrence matrix
    x_ref = librosa.feature.stack_memory(chroma_ref, n_steps=10, delay=3)
    x_comp = librosa.feature.stack_memory(chroma_comp, n_steps=10, delay=3)
    xsim = librosa.segment.cross_similarity(x_comp, x_ref, metric='euclidean', mode='distance')
    xsim_aff = librosa.segment.cross_similarity(x_comp, x_ref, metric='cosine', mode='affinity')

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    imgsim = librosa.display.specshow(xsim, x_axis='s', y_axis='s',
                                      hop_length=hop_length, ax=ax[0])
    ax[0].set(title='Binary cross-similarity (symmetric)')
    imgaff = librosa.display.specshow(xsim_aff, x_axis='s', y_axis='s',
                                      cmap='magma_r', hop_length=hop_length, ax=ax[1])
    ax[1].set(title='Cross-affinity')
    ax[1].label_outer()
    fig.colorbar(imgsim, ax=ax[0], orientation='horizontal', ticks=[0, 1])
    fig.colorbar(imgaff, ax=ax[1], orientation='horizontal')
    fig.show()
    plt.savefig("SimilarityMapping.png")

    # y, sr = librosa.load(filename1 + '.wav')
    # y = librosa.util.normalize(y)
    # n_fft = 1024
    #
    # D1 = librosa.stft(y_ref, n_fft=n_fft)
    # D1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
    # fig, ax = plt.subplots(figsize=(10, 5))
    # librosa.display.specshow(S, ax=ax)
    # S = librosa.feature.melspectrogram(y=y_ref, sr=sr1, n_mels=128,
    #                                    fmax=8000)
    # fig, ax = plt.subplots()
    # S_dB = librosa.power_to_db(S, ref=np.max)
    # img = librosa.display.specshow(S_dB, x_axis='time',
    #                                y_axis='mel', sr=sr1,
    #                                fmax=8000, ax=ax)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram')
    #
    #
    # plt.show()
    # plt.savefig("temp.png")
    #
    # D2 = librosa.stft(y_comp, n_fft=n_fft)
    # D2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)
    # fig, ax = plt.subplots(figsize=(10, 5))
    # librosa.display.specshow(D2, ax=ax)
    #
    # plt.show()
    # plt.savefig("temp2.png")
    # minn = 100000000
    index = 0
    maxx = 0

    for i in range(xsim_aff.shape[0]):
        j = 0
        cur = 0
        while i + j < xsim_aff.shape[0] and j < xsim_aff.shape[1]:
            cur += xsim_aff[i + j][j]
            j += 1
        if maxx < cur / math.sqrt(j):
            maxx = cur / math.sqrt(j)
            index = i
    # for i in range(xsim_aff.shape[1]):
    #     j = 0
    #     cur = 0
    #     while i + j < xsim_aff.shape[1] and j < xsim_aff.shape[0]:
    #         cur += xsim_aff[j][i + j]
    #         j += 1
    #     if maxx < cur:
    #         maxx = cur
    #         index = i
    #
    # for i in range(D2.shape[1] - D1.shape[1]):
    #
    #     # print(mean_squared_error(D2.T[i:D1.shape[1] + i], D1.T))
    #
    #     if minn > mean_squared_error(D2.T[i:D1.shape[1] + i], D1.T):
    #         minn = mean_squared_error(D2.T[i:D1.shape[1] + i], D1.T)
    #         index = i

    clip1 = VideoFileClip(filename1)
    time = AudioFileClip(filename1 + '.wav').duration
    start = index / xsim_aff.shape[0] * time

    if AudioFileClip(filename1).duration > AudioFileClip(filename2).duration:
        end = AudioFileClip(filename2).duration
    else:
        end = AudioFileClip(filename1).duration

    clip1 = clip1.subclip(start, end)
    clip2 = VideoFileClip(filename2)

    if clip1.fps > clip2.fps:
        clip1 = clip1.set_fps(clip2.fps)
    else:
        clip2 = clip2.set_fps(clip1.fps)

    clip1.write_videofile(filename1 + 'edited.mp4')
    clip2.write_videofile(filename2 + 'edited.mp4')

    return filename1 + 'edited.mp4', filename2 + 'edited.mp4', clip1.fps
