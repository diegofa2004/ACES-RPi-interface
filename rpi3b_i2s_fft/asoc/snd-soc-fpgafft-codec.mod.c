#include <linux/module.h>
#include <linux/export-internal.h>
#include <linux/compiler.h>

MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0xfa474811, "__platform_driver_register" },
	{ 0x83d07e6, "_dev_info" },
	{ 0x84f321a, "devm_snd_soc_register_component" },
	{ 0xc00e0644, "snd_pcm_hw_constraint_minmax" },
	{ 0x61fd46a9, "platform_driver_unregister" },
	{ 0xe56a9336, "snd_pcm_format_width" },
	{ 0x474e54d2, "module_layout" },
};

MODULE_INFO(depends, "snd-soc-core,snd-pcm");

MODULE_ALIAS("of:N*T*Caces,fpgafft-codec");
MODULE_ALIAS("of:N*T*Caces,fpgafft-codecC*");

MODULE_INFO(srcversion, "0403DE146EE4B1F2F974CA5");
