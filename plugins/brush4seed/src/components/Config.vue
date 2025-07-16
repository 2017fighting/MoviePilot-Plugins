<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
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
const config = ref({...props.initialConfig})

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
  <VCard  title="刷流保种 - 配置">
      <VDialogCloseBtn @click="emit('close')" />
      <VDivider />
      <VCardText>
        <Form>
          <VRow>
            <VCol>
              <VSelect :required="true" :model="config.downloader_name" :items="config.downloaders" label="下载器"></VSelect>
            </VCol>
          </VRow>
        </Form>
      </VCardText>
      <VCardActions class="pt-3">
        <VBtn @click="emit('switch')" color="info">
          查看数据
        </VBtn>
        <VSpacer />
        <!-- 只有Vuetify模式显示默认保存按钮，Vue模式由组件内部控制 -->
        <VBtn @click="saveConfig" prepend-icon="mdi-content-save" class="px-5">
          保存
        </VBtn>
      </VCardActions>
    </VCard>

</template>