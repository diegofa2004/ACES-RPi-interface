// SPDX-License-Identifier: GPL-2.0
/*
 * Minimal ASoC codec stub for the ACES FPGA -> Raspberry Pi I2S capture path.
 *
 * The FPGA is always the physical I2S master. This driver does not control
 * clocks, parse the wire protocol, or expose any mixer/control surface. Its
 * only job is to provide a capture-only codec-side DAI so simple-audio-card
 * can register a stable ALSA sound card for the raw 32-bit transport.
 */

#include <linux/module.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <sound/pcm.h>
#include <sound/pcm_params.h>
#include <sound/soc.h>

#define FPGAFFT_CODEC_DRIVER_NAME "snd-soc-fpgafft-codec"
#define FPGAFFT_CODEC_DAI_NAME "fpgafft-codec-dai"
#define FPGAFFT_CAPTURE_HOST_RATE_HZ 48828
#define FPGAFFT_CAPTURE_RATE_NUM 390625
#define FPGAFFT_CAPTURE_RATE_DEN 8
#define FPGAFFT_CHANNEL_COUNT 2
#define FPGAFFT_SLOT_WIDTH_BITS 32
#define FPGAFFT_TDM_SLOTS 2

static int fpgafft_codec_startup(struct snd_pcm_substream *substream,
				 struct snd_soc_dai *dai)
{
	struct snd_pcm_runtime *runtime = substream->runtime;
	int ret;

	(void)dai;

	ret = snd_pcm_hw_constraint_single(runtime, SNDRV_PCM_HW_PARAM_RATE,
					   FPGAFFT_CAPTURE_HOST_RATE_HZ);
	if (ret < 0)
		return ret;

	return snd_pcm_hw_constraint_single(runtime, SNDRV_PCM_HW_PARAM_CHANNELS,
					    FPGAFFT_CHANNEL_COUNT);
}

static int fpgafft_codec_hw_params(struct snd_pcm_substream *substream,
				   struct snd_pcm_hw_params *params,
				   struct snd_soc_dai *dai)
{
	(void)substream;
	(void)dai;

	if (params_rate(params) != FPGAFFT_CAPTURE_HOST_RATE_HZ)
		return -EINVAL;

	if (params_channels(params) != FPGAFFT_CHANNEL_COUNT)
		return -EINVAL;

	if (params_format(params) != SNDRV_PCM_FORMAT_S32_LE)
		return -EINVAL;

	if (params_width(params) != FPGAFFT_SLOT_WIDTH_BITS)
		return -EINVAL;

	return 0;
}

static int fpgafft_codec_set_fmt(struct snd_soc_dai *dai, unsigned int fmt)
{
	unsigned int format = fmt & SND_SOC_DAIFMT_FORMAT_MASK;
	unsigned int inversion = fmt & SND_SOC_DAIFMT_INV_MASK;
	unsigned int master = fmt & SND_SOC_DAIFMT_MASTER_MASK;

	(void)dai;

	if (format != SND_SOC_DAIFMT_I2S)
		return -EINVAL;

	if (inversion != SND_SOC_DAIFMT_NB_NF)
		return -EINVAL;

	/*
	 * The clock-master role is declared by simple-audio-card in DT. The
	 * codec stub does not drive pins itself, but we still reject formats
	 * that contradict the fixed FPGA-master topology.
	 */
	if (master != 0 && master != SND_SOC_DAIFMT_CBM_CFM)
		return -EINVAL;

	return 0;
}

static int fpgafft_codec_set_tdm_slot(struct snd_soc_dai *dai,
				      unsigned int tx_mask,
				      unsigned int rx_mask,
				      int slots,
				      int slot_width)
{
	(void)dai;
	(void)tx_mask;
	(void)rx_mask;

	if (slots != FPGAFFT_TDM_SLOTS)
		return -EINVAL;

	if (slot_width != FPGAFFT_SLOT_WIDTH_BITS)
		return -EINVAL;

	return 0;
}

static const struct snd_soc_dai_ops fpgafft_codec_dai_ops = {
	.startup = fpgafft_codec_startup,
	.hw_params = fpgafft_codec_hw_params,
	.set_fmt = fpgafft_codec_set_fmt,
	.set_tdm_slot = fpgafft_codec_set_tdm_slot,
};

static struct snd_soc_dai_driver fpgafft_codec_dai = {
	.name = FPGAFFT_CODEC_DAI_NAME,
	.capture = {
		.stream_name = "FPGA FFT Capture",
		.channels_min = FPGAFFT_CHANNEL_COUNT,
		.channels_max = FPGAFFT_CHANNEL_COUNT,
		.rates = SNDRV_PCM_RATE_CONTINUOUS | SNDRV_PCM_RATE_KNOT,
		.rate_min = FPGAFFT_CAPTURE_HOST_RATE_HZ,
		.rate_max = FPGAFFT_CAPTURE_HOST_RATE_HZ,
		.formats = SNDRV_PCM_FMTBIT_S32_LE,
	},
	.ops = &fpgafft_codec_dai_ops,
};

static const struct snd_soc_component_driver fpgafft_codec_component = {
	.name = FPGAFFT_CODEC_DRIVER_NAME,
};

static int fpgafft_codec_probe(struct platform_device *pdev)
{
	dev_info(&pdev->dev,
		 "registering minimal FPGA FFT codec stub, host rate %u Hz (wire rate %u/%u Hz)\n",
		 FPGAFFT_CAPTURE_HOST_RATE_HZ,
		 FPGAFFT_CAPTURE_RATE_NUM,
		 FPGAFFT_CAPTURE_RATE_DEN);

	return devm_snd_soc_register_component(&pdev->dev,
					       &fpgafft_codec_component,
					       &fpgafft_codec_dai,
					       1);
}

static const struct of_device_id fpgafft_codec_of_match[] = {
	{ .compatible = "aces,fpgafft-codec" },
	{ }
};
MODULE_DEVICE_TABLE(of, fpgafft_codec_of_match);

static struct platform_driver fpgafft_codec_driver = {
	.probe = fpgafft_codec_probe,
	.driver = {
		.name = FPGAFFT_CODEC_DRIVER_NAME,
		.of_match_table = fpgafft_codec_of_match,
	},
};
module_platform_driver(fpgafft_codec_driver);

MODULE_DESCRIPTION("Minimal ASoC codec stub for ACES FPGA FFT I2S capture");
MODULE_AUTHOR("OpenAI Codex");
MODULE_LICENSE("GPL");
