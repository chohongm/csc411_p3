<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

#if user clicks edit of her post.
if (isset($_GET['edit'])){
	$post = $_SESSION['post'];
	$parent = $post['parent'];
	$poster = $post['poster'];
	$title = $post['title'];
	$message = $post['message'];
	$postid = $post['postid'];

#if post being submitted is a duplicate.
} else if (isset($error)){
	$parent = $_POST['parent'];
	$poster = $_POST['poster'];
	$title = $_POST['title'];
	$message = $_POST['message'];
	echo '<span class = spec><br/>*The message you are trying to post is already posted. 
								   Please make sure the content of the post is different 
								   from its original post.<br/><br/></span>';
#if user clicks reply or new post.
} else {
	if (isset($_GET['parent'])){
		$parent = $_GET['parent'];
	} else {
		$parent = 0;
	}
	
	if (!check_valid_user()){
		echo "<p><br/>Users must log-in to submit posts or reply to posts.<br/></p>
				<a href='/user_auth/login.php?url=post_form&parent=$parent'>Click to Log-In</a>";
		do_footer();
		exit;
	}
	
	if (isset($_SESSION['userid'])) {
		$poster = $_SESSION['userid'];
	} else {
		$poster = '';
	}
	$title = 'New Post';
	$message = '';
	
	#if parent is specified (if this is a reply post), set title and message in reply form.
	if ($parent != 0){
		if (strstr($title, 'Re: ') == false){
			$title = 'Re: ';
		}
		if (strlen($title) > 20){
			$title = substr($title, 0, 20);
		}
		$message = add_quoting(get_post_message($parent));
	}
}

do_forum_header($title);

if (isset($_GET['edit'])){
	display_post_form($parent, $poster, $title, $message, $postid);
} else {
	display_post_form($parent, $poster, $title, $message);
}

do_footer();
?>