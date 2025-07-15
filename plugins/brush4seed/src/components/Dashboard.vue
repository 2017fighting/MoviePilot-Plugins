<script setup lang="ts">
// 接收配置和刷新控制
const props = defineProps({
  config: {
    type: Object,
    default: () => ({})
  },
  allowRefresh: {
    type: Boolean,
    default: true
  }
})

// 仪表板逻辑...
</script>

<template>
  <div class="dashboard-widget">
    <!-- 仪表板内容 -->
    <v-card>
      <v-card-title>{{ config.title || '仪表板组件' }}</v-card-title>
      <v-card-text>
        <!-- 组件内容 -->
      </v-card-text>
    </v-card>
  </div>
</template>