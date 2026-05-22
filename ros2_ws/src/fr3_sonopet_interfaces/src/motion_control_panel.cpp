#include <memory>
#include <string>

#include <QHBoxLayout>
#include <QCheckBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

#include "fr3_sonopet_interfaces/action/execute_motion.hpp"
#include "fr3_sonopet_interfaces/action/capture_point_cloud.hpp"
#include "fr3_sonopet_interfaces/action/preview_motion.hpp"
#include "fr3_sonopet_interfaces/action/stop_motion.hpp"
#include "std_srvs/srv/set_bool.hpp"

namespace fr3_sonopet_interfaces
{

class MotionControlPanel final : public rviz_common::Panel
{
public:
  explicit MotionControlPanel(QWidget * parent = nullptr)
  : rviz_common::Panel(parent)
  {
    auto * layout = new QVBoxLayout(this);
    auto * motion_row = new QHBoxLayout();
    auto * operator_row = new QHBoxLayout();
    preview_button_ = new QPushButton("Preview", this);
    execute_button_ = new QPushButton("Execute", this);
    stop_button_ = new QPushButton("Stop Motion", this);
    rescan_button_ = new QPushButton("Rescan Cloud", this);
    save_artifacts_ = new QCheckBox("Save artifacts at session end", this);
    confirmation_ = new QLineEdit(this);
    status_ = new QLabel("Waiting for RViz node.", this);
    confirmation_->setMaxLength(1);
    confirmation_->setPlaceholderText("E");
    save_artifacts_->setChecked(true);
    preview_button_->setEnabled(false);
    execute_button_->setEnabled(false);
    stop_button_->setEnabled(false);
    rescan_button_->setEnabled(false);
    save_artifacts_->setEnabled(false);
    motion_row->addWidget(preview_button_);
    motion_row->addWidget(new QLabel("Confirm:", this));
    motion_row->addWidget(confirmation_);
    motion_row->addWidget(execute_button_);
    motion_row->addWidget(stop_button_);
    operator_row->addWidget(rescan_button_);
    operator_row->addWidget(save_artifacts_);
    layout->addLayout(operator_row);
    layout->addLayout(motion_row);
    layout->addWidget(status_);
    connect(preview_button_, &QPushButton::clicked, this, [this]() {
      PreviewMotion::Goal goal;
      rclcpp_action::Client<PreviewMotion>::SendGoalOptions options;
      status_->setText("Preview requested.");
      options.goal_response_callback = [this](auto goal_handle) {
        status_->setText(goal_handle ? "Preview accepted." : "Preview rejected.");
      };
      options.result_callback = [this](const auto & wrapped_result) {
        status_->setText(QString::fromStdString(wrapped_result.result->message));
      };
      preview_client_->async_send_goal(goal, options);
    });
    connect(execute_button_, &QPushButton::clicked, this, [this]() {
      if (confirmation_->text() != "E") {
        status_->setText("Type E to execute.");
        return;
      }
      ExecuteMotion::Goal goal;
      goal.confirmation_token = "E";
      rclcpp_action::Client<ExecuteMotion>::SendGoalOptions options;
      status_->setText("Execution requested.");
      options.goal_response_callback = [this](auto goal_handle) {
        status_->setText(goal_handle ? "Execution accepted." : "Execution rejected.");
      };
      options.feedback_callback = [this](auto, const auto feedback) {
        status_->setText(QString::fromStdString("Executing: " + feedback->active_segment));
      };
      options.result_callback = [this](const auto & wrapped_result) {
        status_->setText(QString::fromStdString(wrapped_result.result->message));
      };
      execute_client_->async_send_goal(goal, options);
    });
    connect(stop_button_, &QPushButton::clicked, this, [this]() {
      StopMotion::Goal goal;
      rclcpp_action::Client<StopMotion>::SendGoalOptions options;
      status_->setText("Stop requested.");
      options.goal_response_callback = [this](auto goal_handle) {
        status_->setText(goal_handle ? "Stop accepted." : "Stop rejected.");
      };
      options.feedback_callback = [this](auto, const auto feedback) {
        status_->setText(QString::fromStdString("Stopping: " + feedback->phase));
      };
      options.result_callback = [this](const auto & wrapped_result) {
        status_->setText(QString::fromStdString(wrapped_result.result->message));
      };
      stop_client_->async_send_goal(goal, options);
    });
    connect(rescan_button_, &QPushButton::clicked, this, [this]() {
      CapturePointCloud::Goal goal;
      goal.label = "";
      goal.publish_planning_cloud = true;
      goal.save_artifacts = save_artifacts_->isChecked();
      rclcpp_action::Client<CapturePointCloud>::SendGoalOptions options;
      status_->setText("Pointcloud rescan requested.");
      options.goal_response_callback = [this](auto goal_handle) {
        status_->setText(goal_handle ? "Rescan accepted." : "Rescan rejected.");
      };
      options.feedback_callback = [this](auto, const auto feedback) {
        status_->setText(QString::fromStdString("Rescanning: " + feedback->phase));
      };
      options.result_callback = [this](const auto & wrapped_result) {
        status_->setText(QString::fromStdString(wrapped_result.result->message));
      };
      capture_client_->async_send_goal(goal, options);
    });
    connect(save_artifacts_, &QCheckBox::stateChanged, this, [this](int state) {
      auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
      request->data = state == Qt::Checked;
      artifact_saving_client_->async_send_request(
        request,
        [this](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future) {
          status_->setText(QString::fromStdString(future.get()->message));
        });
    });
  }

  void onInitialize() override
  {
    auto rviz_node = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
    preview_client_ = rclcpp_action::create_client<PreviewMotion>(
      rviz_node,
      "/sonopet/preview_motion");
    execute_client_ = rclcpp_action::create_client<ExecuteMotion>(
      rviz_node,
      "/sonopet/execute_motion");
    stop_client_ = rclcpp_action::create_client<StopMotion>(
      rviz_node,
      "/sonopet/stop_motion");
    capture_client_ = rclcpp_action::create_client<CapturePointCloud>(
      rviz_node,
      "/sonopet/capture_pointcloud");
    artifact_saving_client_ = rviz_node->create_client<std_srvs::srv::SetBool>(
      "/sonopet/set_artifact_saving");
    preview_button_->setEnabled(true);
    execute_button_->setEnabled(true);
    stop_button_->setEnabled(true);
    rescan_button_->setEnabled(true);
    save_artifacts_->setEnabled(true);
    status_->setText("Motion controls ready.");
  }

private:
  using PreviewMotion = fr3_sonopet_interfaces::action::PreviewMotion;
  using ExecuteMotion = fr3_sonopet_interfaces::action::ExecuteMotion;
  using StopMotion = fr3_sonopet_interfaces::action::StopMotion;
  using CapturePointCloud = fr3_sonopet_interfaces::action::CapturePointCloud;

  QPushButton * preview_button_;
  QPushButton * execute_button_;
  QPushButton * stop_button_;
  QPushButton * rescan_button_;
  QCheckBox * save_artifacts_;
  QLineEdit * confirmation_;
  QLabel * status_;
  rclcpp_action::Client<PreviewMotion>::SharedPtr preview_client_;
  rclcpp_action::Client<ExecuteMotion>::SharedPtr execute_client_;
  rclcpp_action::Client<StopMotion>::SharedPtr stop_client_;
  rclcpp_action::Client<CapturePointCloud>::SharedPtr capture_client_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr artifact_saving_client_;
};

}  // namespace fr3_sonopet_interfaces

PLUGINLIB_EXPORT_CLASS(fr3_sonopet_interfaces::MotionControlPanel, rviz_common::Panel)
