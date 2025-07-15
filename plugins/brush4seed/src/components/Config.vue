<script setup lang="ts">
// 接收初始配置和API对象
const props = defineProps({
  initialConfig: {
    type: Object,
    default: () => ({})
  },
  api: {
    type: Object,
    default: () => {}
  }
})

// 配置数据
// const config = ref({...props.initialConfig})
const config = props.initialConfig

// 自定义事件，用于保存配置
const emit = defineEmits(['save', 'close', 'switch'])

// 保存配置
function saveConfig() {
  emit('save', config.value)
}

// 通知主应用切换到详情页面
function notifySwitch() {
  emit('switch')
}

// 通知主应用关闭当前页面
function notifyClose() {
  emit('close')
}
</script>

<template>
  <div class="plugin-config">
    <!-- 配置表单示例 -->
    <v-text-field v-model="config.someField" label="配置项"></v-text-field>
    
    <!-- 保存按钮示例 -->
    <v-btn color="primary" @click="saveConfig">保存配置</v-btn>

    <!-- 关闭按钮示例 -->
    <v-btn color="primary" @click="notifyClose">关闭页面</v-btn>

    <!-- 切换按钮示例 -->
    <v-btn color="primary" @click="notifySwitch">切换到详情页面</v-btn>
  </div>
</template>