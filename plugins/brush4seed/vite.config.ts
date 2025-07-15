import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import federation from "@originjs/vite-plugin-federation";
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
    plugins: [
        vue(),
        federation({
            name: 'remote-app',
            filename: 'remoteEntry.js',
            // 需要暴露的模块
            exposes: {
                './Page': './src/components/Page.vue',
                './Config': './src/components/Config.vue',
                './Dashboard': './src/components/Dashboard.vue',
            },
            shared: ['vue', 'vuetify'],
        }),
        topLevelAwait({
            // The export name of top-level await promise for each chunk module
            promiseExportName: "__tla",
            // The function to generate import names of top-level await promise in each chunk module
            promiseImportName: i => `__tla_${i}`
        })
    ],
    build: {
        // target: 'esnext',   // 必须设置为esnext以支持顶层await
        minify: false,      // 开发阶段建议关闭混淆
        cssCodeSplit: true, // 改为true以便能分离样式文件
    },
    css: {
        preprocessorOptions: {
            scss: {
                additionalData: '/* 覆盖vuetify样式 */',
            }
        },
        postcss: {
            plugins: [
                {
                    postcssPlugin: 'internal:charset-removal',
                    AtRule: {
                        charset: (atRule) => {
                            if (atRule.name === 'charset') {
                                atRule.remove();
                            }
                        }
                    }
                },
                {
                    postcssPlugin: 'vuetify-filter',
                    Root(root) {
                        // 过滤掉所有vuetify相关的CSS
                        root.walkRules(rule => {
                            if (rule.selector && (
                                rule.selector.includes('.v-') ||
                                rule.selector.includes('.mdi-'))) {
                                rule.remove();
                            }
                        });
                    }
                }
            ]
        }
    },
    server: {
        port: 5001,   // 使用不同于主应用的端口
        cors: true,   // 启用CORS
        origin: 'http://localhost:5001'
    },
}) 
