# SCSS结构说明

## 目录整合

本项目SCSS文件已完成整合：
- 主入口文件：`src/@core/scss/index.scss`
- 实际功能文件位于：`src/@core/scss/template/index.scss`

## 整合内容

- 整合了原`src/@core/scss/base`和`src/@core/scss/template`目录的功能
- 统一使用`template`目录作为SCSS样式的主要引用点
- 保留原有引用结构以保证向后兼容性

## 整合进度

已完成：
- ✅ 主入口文件引用更新
- ✅ mixins文件合并
- ✅ placeholders目录下文件转移
- ✅ perfect-scrollbar文件整合
- ✅ vuetify相关文件整合
- ✅ default-layout-w-vertical-nav文件整合
- ✅ 移除了template/index.scss中对base目录组件的依赖
- ✅ 修复了components.scss中对base/mixins的引用
- ✅ 修复了variables.scss中对base/variables的引用
- ✅ 修复了apex-chart.scss和full-calendar.scss的linter错误
- ✅ 整合并移除了对vuetify/variables的依赖
- ✅ 修复了SCSS变量名冲突问题
- ✅ 修复了SASS模块重复加载配置问题
- ✅ 修复了导入路径问题（misc、utils等模块的引用路径）

待完成：
- ⬜ 最终测试确保无样式问题
- ⬜ 清理冗余文件

## 使用方式

在项目中引用SCSS时，应使用：
```scss
@use "@core/scss";
```

这将自动加载所有必要的样式文件。

## 注意事项

此次整合已将所有功能文件整合到template目录，不再依赖base目录的代码。现在可以安全地从外部引用template目录下的文件，但需要进行最终测试以确保样式正常工作。

测试无误后，可以考虑完全删除base目录，以简化项目结构。

## 最近修复

在最近的更新中，我们修复了以下问题：
1. 解决了变量名冲突问题，通过使用命名空间（如`layouts-vars`）来引用外部模块变量
2. 修复了SASS模块重复配置问题，将多处的`@forward...with`配置合并到了template/_variables.scss文件中
3. 统一使用命名空间引用模块，避免后续出现冲突
4. 修复了`_default-layout-w-vertical-nav.scss`中导入路径错误，将`@use "misc"`修改为`@use "../misc"`
