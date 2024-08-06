Band4Depth4 = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]  # patch_size=5,window_size=25时，4层网络允许的光谱通道数
Band4Depth3 = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]    # patch_size=5,window_size=25时，3层网络允许的光谱通道数
# (145,145,200)
IN_PATH = r'hsi_data/Indian_Pines/Indian_pines_corrected.mat'
IN_GT_PATH = r'hsi_data/Indian_Pines/Indian_pines_gt.mat'
# (1096, 715, 102)
PA = r'./hsi_data/pavia/Pavia.mat'
PA_GT = r'./hsi_data/pavia/Pavia_gt.mat'
# (610, 340, 103) *
PU = r'./hsi_data/paviaU/PaviaU.mat'
PU_GT = r'./hsi_data/paviaU/PaviaU_gt.mat'
PU_LABELS = [
            "Undefined",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]
# (512, 217, 204) *
SA = r'./hsi_data/Salinas/Salinas_corrected.mat'
SA_GT = r'./hsi_data/Salinas/Salinas_gt.mat'
# (83, 86, 204)
SA_A = r'./hsi_data/SalinasA/SalinasA_corrected.mat'
SA_A_GT = r'./hsi_data/SalinasA/SalinasA_gt.mat'
# (349,1905,144)
HOU = 'hsi_data/Houston2013/HSI.mat'
HOU_GT = 'hsi_data/Houston2013/gt.mat'
# (2384,601,48*)
HOU2018 = r'hsi_data/houston2018/HoustonU.mat'
HOU2018_GT = r'hsi_data/houston2018/HoustonU_gt.mat'

__IN = r'hsi_data/corrected/Indian/img.mat'
__IN_gt = r'hsi_data/corrected/Indian/GroundT.mat'

__PU = r'hsi_data/corrected/PaviaU/img.mat'
__PU_gt = r'hsi_data/corrected/PaviaU/GroundT.mat'

__SA = r'hsi_data/corrected/Salinas/img.mat'
__SA_gt = r'hsi_data/corrected/Salinas/GroundT.mat'

# (3750,1580,256)
XiongAn = r'hsi_data/XiongAn/xiongan.mat'
XIongAn_GT = r'hsi_data/XiongAn/xiongan_gt.mat'
# (500,260,436)
Xuzhou = r'hsi_data/Xuzhou/xuzhou.mat'
Xuzhou_GT = r'hsi_data/Xuzhou/xuzhou_gt.mat'
XZ_LABELS = [
            'Undefined',
            'BARELAND1',
            'LAKES',
            'COALS',
            'CEMENT',
            'CROPS-1',
            'TRESS',
            'BARELAND2',
            'CROPS',
            'RED-TITLE'
        ]
# (512,614,176)
KSC = r'hsi_data/KSC/KSC.mat'
KSC_GT = r'hsi_data/KSC/KSC_gt.mat'
# (1476, 256, 145)
Botswana = r'hsi_data/Botswana/Botswana.mat'
Botswana_GT = r'hsi_data/Botswana/Botswana_gt.mat'
# (1217, 303, 274),16
WHU_Hi_HanChuan = r'hsi_data/WHU_Hi_HanChuan/WHU_Hi_HanChuan.mat'
WHU_Hi_HanChuan_GT = r'hsi_data/WHU_Hi_HanChuan/WHU_Hi_HanChuan_gt.mat'
# (940, 475, 270),22
WHU_Hi_HongHu = r'hsi_data/WHU_Hi_HongHu/WHU_Hi_HongHu.mat'
WHU_Hi_HongHu_GT = r'hsi_data/WHU_Hi_HongHu/WHU_Hi_HongHu_gt.mat'
# (550, 400, 270),9
WHU_Hi_LongKou = r'hsi_data/WHU_Hi_LongKou/WHU_Hi_LongKou.mat'
WHU_Hi_LongKou_GT = r'hsi_data/WHU_Hi_LongKou/WHU_Hi_LongKou_gt.mat'
LK_LABELS = [
            'Undefined',
            'Corn',
            'Cotton',
            'Sesame',
            'Broad-leaf soybean',
            'Narrow-leaf soybean',
            'Rice',
            'Water',
            'Roads and houses',
            'Mixed weed',
        ]